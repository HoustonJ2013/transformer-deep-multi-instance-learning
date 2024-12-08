import argparse
import json
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataloader import (MnstBagsGenerator,
                        NumpyDataset, NumpyGenerator)
from modelMIL import MILAttention, TransformerMIL
from torchmetrics.classification import BinaryF1Score
from utils import load_config, save_config, save_log


torch.multiprocessing.set_sharing_strategy('file_descriptor')
cudnn.benchmark = True

def get_args_parser():
    parser = argparse.ArgumentParser("tdmil", add_help=False)

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
        "--eval_frequency",
        default=4,
        type=int,
        help="""run evaluation for every n step""",
    )
    
    parser.add_argument(
        "--load_from_checkpoint",
        action="store_false",
        help="Continue training from the existing checkpoint from the checkpoint folder",
    )
    parser.add_argument(
        "--binary_classification",
        action="store_false",
        help="If the task is binary_classification",
    )
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        help="the configure file for the test",
    )

    # parameter for data
    parser.add_argument(
        "--emb_size",
        default=384,
        type=int,
        help="""embedding size""",
    )
    parser.add_argument(
        "--loader_type",
        default="MnstBagsGenerator",
        type=str,
        choices=["MnstBagsGenerator", "NumpyGenerator"],
        help="""the type of dataloader""",
    )
    parser.add_argument(
        "--mnst_train_inp_csv",
        default="datasets/mnst_train.csv",
        type=str,
        help="""The path to the csv contain mnst train data""",
    )
    parser.add_argument(
        "--mnst_test_inp_csv",
        default="datasets/mnst_test.csv",
        type=str,
        help="""The path to the csv contain mnst test data""",
    )

    parser.add_argument(
        "--train_inp_csv",
        default="datasets/mnst_train.csv",
        type=str,
        help="""The path to the csv contain train data""",
    )
    parser.add_argument(
        "--test_inp_csv",
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
        "--mnst_target",
        default=9,
        type=int,
        help="""the target number for the mnst classifier""",
    )
    parser.add_argument(
        "--mnst_target_multiples",
        default=1,
        type=int,
        help="""the min number of target numbers show up in the bag to consider as bag as 1""",
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
    # Parameter for MIL attention
    parser.add_argument(
        "--mil_middle_dim",
        default=128,
        type=int,
        help="""The second dimension of V in the attention layer""",
    )
    parser.add_argument(
        "--mil_attention_branches",
        default=1,
        type=int,
        help="""The second dimension of W in the attention layer""",
    )

    parser.add_argument(
        "--transformer_depth",
        default=1,
        type=int,
        help="""depth of the transformer""",
    )

    parser.add_argument(
        "--transformer_num_heads",
        default=12,
        type=int,
        help="""number of transformer heads""",
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
        default=0.001,
        help="""learning rate""",
    )
    parser.add_argument(
        "--loss",
        default="BCEWithLogitsLoss",
        type=str,
        choices=["BCEWithLogitsLoss", "CrossEntropy", "MAE", "MSE"],
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


def save_checkpoint(model, optimizer, epoch, loss, args):
    test_folder = os.path.join(args.checkpoint_path, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    filename = "checkpoint_%05d.pt" % (epoch)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        os.path.join(test_folder, filename),
    )
    return


def load_checkpoint(model, optimizer, args):
    test_folder = os.path.join(args.checkpoint_path, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    all_files = [_ for _ in os.listdir(test_folder) if "checkpoint" in _]
    if len(all_files) == 0:
        return 0, 0
    filename_lastest = sorted(all_files)[-1]
    checkpoint = torch.load(os.path.join(test_folder, filename_lastest))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    return epoch, loss


def train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch, n_batches, args, refresh_freq=2):

    loss_values = []
    ## When the attention_model is light weighted, most of the time is on I/O, copying the data
    ## from cpu to GPU.
    refresh_freq = max([1, int(n_batches * 0.01)])
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader.dataloader(random_seed=epoch)):
        if batch_i % refresh_freq == 0:
            print(f"step {batch_i}/{n_batches} at epoch {epoch} with loss {np.mean(loss_values):0.2e} \r", end="", flush=True)
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
    print()
    print(f"Train loss at epoch {epoch} is {np.mean(loss_values):0.2e}")
    train_metrics = {"loss": np.mean(loss_values)}
    return train_metrics


def eval_model(attention_model, test_loader, loss_fn, epoch, n_batches, args, binary_classification=True):
    true_labels = []
    pred_labels = []
    loss_values = []
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
            loss = loss_fn(logits, label_)
            loss_values.append(loss.item())
            if binary_classification:
                y_prob = nn.Sigmoid()(logits)
                y_pred = torch.ge(y_prob, 0.5).float()
                pred_labels.append(y_pred)
    if binary_classification:
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
    else: 
        print("eval loss at epoch %i is %0.2e" % (epoch, np.mean(loss_values)))
              
    eval_metrics = {
        "loss": np.mean(loss_values),
    }
    return eval_metrics


def explain_mnst_model(attention_model, test_loader, epoch, args):
    test_folder = os.path.join(args.checkpoint_path, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    output_path = os.path.join(test_folder, "test_explain.pt")
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
            target_number=args.mnst_target,
            bag_length_dist=args.mnst_bag_length_dist,
            max_bag_length=args.max_bag_length,
            mean_bag_length=args.mnst_mean_bag_length,
            var_bag_length=args.mnst_var_bag_length,
            num_bag=args.num_bags_train,
            target_multiples=args.mnst_target_multiples,
        )

        test_loader = MnstBagsGenerator(
            embedding_tensor_path=args.mnst_test_inp_path,
            label_tensor_path=args.mnst_test_label_path,
            batch_size=64,
            target_number=args.mnst_target,
            bag_length_dist="normal",
            max_bag_length=args.max_bag_length,
            mean_bag_length=args.mnst_mean_bag_length,
            var_bag_length=args.mnst_var_bag_length,
            num_bag=500,
            target_multiples=args.mnst_target_multiples,
        )
        n_batches = len(train_loader)
    elif args.loader_type == "NumpyGenerator":
        train_dataset = NumpyDataset(inp_csv=args.train_inp_csv, max_bag_length=args.max_bag_length)
        train_loader = NumpyGenerator(
            train_dataset, 
            batch_size=args.batch_size, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=True, 
            drop_last=True
        )
        test_dataset = NumpyDataset(inp_csv=args.test_inp_csv, 
                                    max_bag_length=args.test_max_bag_length)
        test_loader = NumpyGenerator(test_dataset, 
                                     batch_size=args.batch_size, 
                                     num_workers=4, 
                                     shuffle=False)
        n_batches = len(train_loader)
    else:
        raise ValueError("The loader type is not supported")
    
    ## define model, optimizer and loss
    if args.arch == "MILAttention":
        attention_model = MILAttention(
            embedding_size=args.emb_size, middle_dim=args.mil_middle_dim, attention_branches=args.mil_attention_branches
        )
    elif args.arch == "transformer":
        attention_model = TransformerMIL(
            embed_dim=args.emb_size,
            depth=args.transformer_depth,
            num_heads=args.transformer_num_heads,
        )
    else:
        attention_model = MILAttention(
            embedding_size=args.emb_size, middle_dim=args.mil_middle_dim, attention_branches=args.mil_attention_branches
        )

    if args.gpu:
        attention_model = attention_model.to("cuda")

    optimizer = torch.optim.Adam(attention_model.parameters(), lr=args.lr)

    if args.loss == "BCEWithLogitsLoss":
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "MAE":
        loss_fn = nn.L1Loss()
    elif args.loss == "MSE":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("The loss function is not supported")
    
    ## start training
    if args.load_from_checkpoint:
        start_epoch, loss_val = load_checkpoint(attention_model, optimizer, args)
        epoch = start_epoch
    else:
        start_epoch = 0
        epoch = start_epoch
        loss_val = 0
    
    for epoch in range(start_epoch, args.num_epoch):
        loss_val = train_one_epoch(attention_model, optimizer, train_loader, loss_fn, epoch, n_batches, args)["loss"]
        eval_loss = None
        if epoch % args.eval_frequency == 0:
            eval_metrics = eval_model(attention_model, 
                                          test_loader, 
                                          loss_fn, 
                                          epoch, 
                                          n_batches, 
                                          args,
                                          binary_classification=args.binary_classification)
            eval_loss = eval_metrics["loss"]
        if (epoch + 1) % args.save_checkpoint_epoch == 0:
            save_checkpoint(attention_model, optimizer, epoch + 1, loss_val, args)

        log_stats = {
            "epoch": epoch,
            "train_loss": loss_val,
            "eval_loss": eval_loss,
        }
        save_log(log_stats, args)

    save_checkpoint(attention_model, optimizer, epoch + 1, loss_val, args)
    eval_metrics = eval_model(attention_model, 
                              test_loader, 
                              loss_fn, 
                              epoch, 
                              n_batches, 
                              args, 
                              binary_classification=args.binary_classification)
    if args.loader_type == "MnstBagsGenerator":
        explain_mnst_model(attention_model, test_loader, epoch, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tdmil", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config_file is not None:
        load_config(args, args.config_file)
    save_config(args)
    print(args)
    train(args)
