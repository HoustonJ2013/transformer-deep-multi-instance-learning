import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MnistBagsGenerator
from modelMIL import MILAttention
from torchmetrics.classification import BinaryF1Score


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="MILAttention",
        type=str,
        choices=["MILAttention", "MILGatedAttention", "transformer"],
        help="""Name of the attention architecture""",
    )
    parser.add_argument(
        "--test_name",
        default="vanilla",
        type=str,
        help="""The name for the trainning test""",
    )
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        help="the configure file to rerun the test",
    )

    # parameter for data
    parser.add_argument(
        "--mnst_train_inp_path",
        default="datasets/mnst_train_dinov2_small.pt",
        type=str,
        help="""The path to the mnst train input embedding""",
    )
    parser.add_argument(
        "--mnst_train_label_path",
        default="datasets/mnst_train_labels.pt",
        type=str,
        help="""The path to the mnst train label tensfor""",
    )

    parser.add_argument(
        "--mnst_test_inp_path",
        default="datasets/mnst_test_dinov2_small.pt",
        type=str,
        help="""The path to the mnst test input embedding""",
    )
    parser.add_argument(
        "--mnst_test_label_path",
        default="datasets/mnst_test_labels.pt",
        type=str,
        help="""The path to the mnst test label tensfor""",
    )
    parser.add_argument(
        "--mnst_bag_length_dist",
        default="poisson",
        type=str,
        choices=["poisson", "normal"],
        help="""the random bag length distribution""",
    )
    parser.add_argument(
        "--mnst_max_bag_length",
        default=20,
        type=int,
        help="""num of bags in the training set""",
    )
    parser.add_argument(
        "--mnst_mean_bag_length",
        default=8,
        type=int,
        help="""num of bags in the training set""",
    )
    parser.add_argument(
        "--mnst_var_bag_length",
        default=5,
        type=int,
        help="""num of bags in the training set""",
    )
    parser.add_argument(
        "--num_bags_train",
        default=1000,
        type=int,
        help="""num of bags in the training set""",
    )
    parser.add_argument(
        "--mnst_load_to_gpu",
        action="store_false",
        help="Directly load MNST data to GPU and sample from there",
    )

    # Parameter for MIL attention
    parser.add_argument(
        "--middle_dim",
        default=128,
        type=int,
        help="""The second dimension of V in the attention layer""",
    )
    parser.add_argument(
        "--attention_branches",
        default=1,
        type=int,
        help="""The second dimension of W in the attention layer""",
    )

    # Basic training parameters
    parser.add_argument("-gpu", action="store_false", help="strongly recommend to use GPU")
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="""batch size""",
    )
    parser.add_argument(
        "--num_epoch",
        default=100,
        type=int,
        help="""num of epochs""",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="""learning rate""",
    )
    parser.add_argument(
        "--loss",
        default="BCEWithLogitsLoss",
        type=str,
        choices=["BCEWithLogitsLoss", "CrossEntropy"],
        help="""The name of the loss""",
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        choices=["Adam"],
        help="""The name of the optimizer""",
    )
    return parser


def train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch, refresh_freq=50):

    loss_values = []
    ## When the attention_model is light weighted, most of the time is on I/O, copying the data
    ## from cpu to GPU.
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader.dataloader(random_seed=epoch)):
        if batch_i % refresh_freq == 0:
            print("step %i at epoch %i \r" % (batch_i, epoch), end="", flush=True)
        if args.gpu:
            inp_ = inp_.cuda(non_blocking=True)
            label_ = label_.cuda(non_blocking=True)
            mask_ = mask_.cuda(non_blocking=True)
        optimizer.zero_grad()
        forward_result = attention_model(inp_, mask_)
        logits = forward_result["logits"].squeeze(-1)
        loss = loss_fn(logits, label_)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    print("loss at epoch %i is %f" % (epoch, np.mean(loss_values)))


def eval_model(attention_model, test_loader, epoch):
    true_labels = []
    pred_labels = []
    for batch_i, (inp_, label_, mask_) in enumerate(test_loader.dataloader()):
        if args.gpu:
            inp_ = inp_.cuda(non_blocking=True)
            mask_ = mask_.cuda(non_blocking=True)
            label_ = label_.cuda(non_blocking=True)
        true_labels.append(label_)
        with torch.no_grad():
            forward_result = attention_model(inp_, mask_)
            logits = forward_result["logits"].squeeze(-1)
            y_prob = nn.Sigmoid()(logits)
            y_pred = torch.ge(y_prob, 0.5).float()
            pred_labels.append(y_pred)
    true_labels = torch.concat(true_labels)
    pred_labels = torch.concat(pred_labels)
    metric = BinaryF1Score()
    f1 = metric(1 - pred_labels.to("cpu"), 1 - true_labels.to("cpu"))
    num_pos = torch.sum(true_labels == 1).cpu()
    print(
        "There are %i test samples, %i positive and %i negative"
        % (len(true_labels), num_pos, len(true_labels) - num_pos)
    )
    print(
        "At epoch %i the accuracy is %f and f1 is %f"
        % (
            epoch,
            pred_labels.eq(true_labels).cpu().float().mean().data.item(),
            f1.float().data.item(),
        )
    )


def explain_model(attention_model, test_loader, epoch, output_path="datasets/test_explain.pt"):
    true_labels = []
    pred_labels = []
    batch_indices = []
    attention_scores = []
    masks = []
    for batch_i, (inp_, label_, mask_, batch_indice) in enumerate(test_loader.dataloader(return_indices=True)):
        if args.gpu:
            inp_ = inp_.cuda(non_blocking=True)
            mask_ = mask_.cuda(non_blocking=True)
            label_ = label_.cuda(non_blocking=True)
        true_labels.append(label_)
        masks.append(mask_)
        with torch.no_grad():
            forward_result = attention_model(inp_, mask_, return_attention=True)
            logits = forward_result["logits"].squeeze(-1)
            attention_score = forward_result["attention"]
            y_prob = nn.Sigmoid()(logits)
            y_pred = torch.ge(y_prob, 0.5).float()
            pred_labels.append(y_pred)
        batch_indices.append(batch_indice)
        attention_scores.append(attention_score)
    true_labels = torch.concat(true_labels)
    pred_labels = torch.concat(pred_labels)
    batch_indices = torch.concat(batch_indices)
    masks = torch.concat(masks)
    attention_scores = torch.concat(attention_scores)
    test_results = {
        "true_label": true_labels.to("cpu"),
        "pred_label": pred_labels.to("cpu"),
        "masks": masks.to("cpu"),
        "batch_indices": batch_indices.to("cpu"),
        "attention_scores": attention_scores.to("cpu"),
    }
    torch.save(test_results, output_path)


def train(args):

    ## define dataloader
    train_loader = MnistBagsGenerator(
        embedding_tensor_path=args.mnst_train_inp_path,
        label_tensor_path=args.mnst_train_label_path,
        batch_size=args.batch_size,
        target_number=9,
        bag_length_dist=args.mnst_bag_length_dist,
        max_bag_length=args.mnst_max_bag_length,
        mean_bag_length=args.mnst_mean_bag_length,
        var_bag_length=args.mnst_var_bag_length,
        num_bag=args.num_bags_train,
        load_to_gpu=args.mnst_load_to_gpu,
    )

    test_loader = MnistBagsGenerator(
        embedding_tensor_path=args.mnst_test_inp_path,
        label_tensor_path=args.mnst_test_label_path,
        batch_size=64,
        target_number=9,
        bag_length_dist="normal",
        max_bag_length=25,
        mean_bag_length=12,
        var_bag_length=5,
        num_bag=500,
        load_to_gpu=args.mnst_load_to_gpu,
    )

    ## define model, optimizer and loss
    attention_model = MILAttention()
    if args.gpu:
        attention_model = attention_model.to("cuda")
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=args.lr)
    if args.loss == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    for epoch in range(start_epoch, args.num_epoch):
        train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch)
        if epoch % 5 == 0:
            eval_model(attention_model, test_loader, epoch)
    eval_model(attention_model, test_loader, epoch)
    explain_model(attention_model, test_loader, epoch, output_path="datasets/test_explain.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tdmil", parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    train(args)
