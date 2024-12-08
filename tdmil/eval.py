import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataloader import (NumpyDataset, NumpyGenerator)
from modelMIL import MILAttention, TransformerMIL


def get_args_parser():
    parser = argparse.ArgumentParser("tdmil", add_help=False)

    parser.add_argument(
        "--checkpoint_path",
        default="model_weights/mnst_mil_vallina/",
        type=str,
        help="""The folder to save model weights""",
    )
    parser.add_argument(
        "--inp_csv",
        default="datasets/train_image_emb.csv",
        type=str,
        help="""The path to the csv contain image embeddings""",
    )

    parser.add_argument(
        "--output_format",
        default="pandas_pkl",
        type=str,
        choices=["pandas_pkl", "numpy", "torch"],
        help="""the format for the output file""",
    )

    parser.add_argument(
        "--num_rerun",
        default=1,
        type=int,
        help="""When the number of images in a bag is more than the max bag length,
        rerun the bag generation process will increase the chance of all images being selected""",
    )

    # parser.add_argument(
    #     "--output_content",
    #     default="cls_embedding",
    #     type=str,
    #     choices=["cls_embedding", "all_embeddings"],
    #     help="""the format for the output file""",
    # )
    parser.add_argument("--gpu", action="store_false", help="strongly recommend to use GPU")
    return parser


def eval_model(attention_model, test_loader, args, checkpoint_config):
    loss_fn = nn.L1Loss()
    embeddings = []
    loss_values = []
    for _, (inp_, label_, mask_) in enumerate(test_loader.dataloader()):
        if args.gpu:
            inp_ = inp_.cuda(non_blocking=True)
            mask_ = mask_.cuda(non_blocking=True)
            label_ = label_.cuda(non_blocking=True)
        with torch.no_grad():
            if checkpoint_config["arch"] == "MILAttention":
                forward_result = attention_model(inp_, mask_, return_feature=True)
                batch_embedding = forward_result["feature"].clone().detach().cpu().squeeze().numpy().astype(np.float32)
            elif checkpoint_config["arch"] == "transformer":
                forward_result = attention_model(inp_, mask_, return_cls_token=True)
                batch_embedding = forward_result["cls_token"].clone().detach().cpu().numpy().astype(np.float32)
            else:
                ValueError("model not supported")
            embeddings.extend(list(batch_embedding))
            logits = forward_result["logits"].squeeze(-1)
            loss = loss_fn(logits, label_)
            loss_values.append(loss.item())
    print("eval loss is %0.2e" % (np.mean(loss_values)))
    return embeddings


def load_checkpoint(model, model_path):

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])


def model_eval(args):
    checkpoint_path = args.checkpoint_path
    config_path = os.path.join(checkpoint_path, "config.json")
    all_model_paths = sorted([os.path.join(args.checkpoint_path, f) for f in os.listdir(checkpoint_path)
                       if f.endswith(".pt")])
    model_path = all_model_paths[-1]
    checkpoint_config = json.load(open(config_path))

    if checkpoint_config["arch"] == "MILAttention":
        attention_model = MILAttention(
            embedding_size=checkpoint_config["emb_size"], 
            middle_dim=checkpoint_config["mil_middle_dim"], 
            attention_branches=checkpoint_config["mil_attention_branches"]
        )
    elif checkpoint_config["arch"] == "transformer":
        attention_model = TransformerMIL(
            embed_dim=checkpoint_config["emb_size"],
            depth=checkpoint_config["transformer_depth"],
            num_heads=checkpoint_config["transformer_num_heads"],
        )
    else:
        attention_model = MILAttention(
            embedding_size=args.emb_size, middle_dim=args.mil_middle_dim, attention_branches=args.mil_attention_branches
        )
    if args.gpu:
        attention_model = attention_model.to("cuda")
        
    # load checkpoint
    load_checkpoint(attention_model, model_path)

    test_dataset = NumpyDataset(inp_csv=args.inp_csv, 
                                    max_bag_length=checkpoint_config["max_bag_length"])
    test_loader = NumpyGenerator(test_dataset, 
                                    batch_size=checkpoint_config["batch_size"], 
                                    num_workers=4, 
                                    shuffle=False)
    df = pd.read_csv(args.inp_csv).reset_index(drop=True)
    for i in range(args.num_rerun):
        embeddings = eval_model(attention_model, test_loader, args, checkpoint_config)
        if i == 0:
            df["embedding"] = embeddings
        else:
            df["embedding" + str(i)] = embeddings
    
    output_file = args.inp_csv[:-4] + "_" + checkpoint_config["test_name"] + "_rerun" + str(args.num_rerun)
    if args.output_format == "pandas_pkl":
        df.to_pickle(output_file + ".pkl")
    elif args.output_format == "numpy":
        np.save(output_file + ".npy", embeddings)
    elif args.output_format == "torch":
        torch.save(embeddings, output_file + ".pt")
    else:
        ValueError("output format not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tdmil_eval", parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    model_eval(args)
