import json
import os

import torch


def save_log(log_stats, args):
    test_folder = os.path.join(args.checkpoint_path, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    log_file = os.path.join(test_folder, "log.txt")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_stats) + "\n")


def save_config(args):
    test_folder = os.path.join(args.checkpoint_path, args.test_name)
    if os.path.exists(test_folder) is False:
        os.makedirs(test_folder, exist_ok=True)
    config_file = os.path.join(test_folder, "config.json")
    with open(config_file, "w") as f:
        json.dump(args.__dict__, f, indent=2)


def load_config(args, filename):
    with open(filename, "r") as f:
        args.__dict__.update(json.load(f))
