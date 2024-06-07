import argparse

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MnstBagsGenerator, NumpyConcurrentGenerator, NumpyDataset, NumpyGenerator
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
        "--checkpoint_path",
        default="model_weights/",
        type=str,
        help="""The folder to save model weights""",
    )
    parser.add_argument(
        "--save_checkpoint_epoch",
        default=4,
        type=int,
        help="""save checkpoint for every n step""",
    )
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        help="the configure file to rerun the test",
    )

    # parameter for data

    parser.add_argument(
        "--loader_type",
        default="MnstBagsGenerator",
        type=str,
        choices=["MnstBagsGenerator", "NumpyGenerator", "NumpyConcurrentGenerator"],
        help="""the type of dataloader""",
    )
    parser.add_argument(
        "--mnst_train_inp_csv",
        default="datasets/mnst_train.csv",
        type=str,
        help="""The path to the csv contain train data""",
    )
    parser.add_argument(
        "--mnst_test_inp_csv",
        default="datasets/mnst_test.csv",
        type=str,
        help="""The path to the csv contain test data""",
    )

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
        "--max_bag_length",
        default=40,
        type=int,
        help="""num of bags in the training set""",
    )
    parser.add_argument(
        "--test_max_bag_length",
        default=40,
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


def save_checkpoint(model, optimizer, epoch, loss, args){
    test_folder = os.path.join(args.model_weights, args.test_name)
    filename = "checkpoint_%05d.pt"%(epoch)
    
    torch.save({"epoch": epoch, 
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,}, os.path.join(test_folder, filename))
}


def train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch, n_batches, refresh_freq=2):

    loss_values = []
    ## When the attention_model is light weighted, most of the time is on I/O, copying the data
    ## from cpu to GPU.
    refresh_freq = max([1, int(n_batches * 0.01)])
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader.dataloader(random_seed=epoch)):
        if batch_i % refresh_freq == 0:
            print("step %i/%i at epoch %i \r" % (batch_i, n_batches, epoch), end="", flush=True)
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
    return np.mean(loss_values)


def eval_model(attention_model, test_loader, epoch, n_batches):
    true_labels = []
    pred_labels = []
    refresh_freq = max([1, int(n_batches * 0.01)])
    for batch_i, (inp_, label_, mask_) in enumerate(test_loader.dataloader()):
        if batch_i % refresh_freq == 0:
            print("step %i/%i at epoch %i \r" % (batch_i, n_batches, epoch), end="", flush=True)
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
    f1 = metric(pred_labels.to("cpu"), true_labels.to("cpu"))
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
    if args.loader_type == "MnstBagsGenerator":
        train_loader = MnstBagsGenerator(
            embedding_tensor_path=args.mnst_train_inp_path,
            label_tensor_path=args.mnst_train_label_path,
            batch_size=args.batch_size,
            target_number=9,
            bag_length_dist=args.mnst_bag_length_dist,
            max_bag_length=args.max_bag_length,
            mean_bag_length=args.mnst_mean_bag_length,
            var_bag_length=args.mnst_var_bag_length,
            num_bag=args.num_bags_train,
            load_to_gpu=args.mnst_load_to_gpu,
        )

        test_loader = MnstBagsGenerator(
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
        n_batches = len(train_loader)
    elif args.loader_type == "NumpyGenerator":
        train_dataset = NumpyDataset(inp_csv=args.mnst_train_inp_csv, max_bag_length=args.max_bag_length)
        train_loader = NumpyGenerator(
            train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True
        )

        test_dataset = NumpyDataset(inp_csv=args.mnst_test_inp_csv, max_bag_length=args.test_max_bag_length)
        test_loader = NumpyGenerator(
            test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=True
        )
        n_batches = len(train_loader)
    else:
        train_loader = NumpyConcurrentGenerator(args.mnst_train_inp_csv, batch_size=args.batch_size, max_threads=1024)
        test_loader = NumpyConcurrentGenerator(args.mnst_test_inp_csv, batch_size=args.batch_size, max_threads=1024)
        n_batches = (len(train_loader) + args.batch_size - 1) // (args.batch_size)
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
    loss_val = 0
    for epoch in range(start_epoch, args.num_epoch):
        loss_val = train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch, n_batches)
        if epoch % 5 == 0:
            eval_model(attention_model, test_loader, epoch, n_batches)
        if epoch > 0 and epoch % args.save_checkpoint_epoch == 0:
            save_checkpoint(attention_model, optimizer, epoch, loss_val, args) 
    
    eval_model(attention_model, test_loader, epoch, n_batches)
    explain_model(attention_model, test_loader, epoch, output_path="datasets/test_explain.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tdmil", parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    train(args)
