"""This data loader is inspired by https://github.com/AMLab-Amsterdam/AttentionDeepMIL

    The input data into the model training and inference should have the shape as B x N_seq x E
    B : batch size
    N_seq : the maximum length of the bag, if the number of samples in a bag is less than N_seq, the pad with mask embedding
    E : embedding size for the image/text/other object

    For simulated dataloader, e.g. bags of the MNST data, we have the embedding already saved on disk, following the strategy in the AttentionDeepMIL to generate random bags and label. 
    We modified the model architecture to support batch, so we use batch generator to generate random bags

    For the real dataset, the embedding should be applied to the images first and saved on the disk. The dataloader will read the data sample and pad with zeros to the max length 
    if the length of the sample is less than max bag length. 

"""

from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, as_completed, wait
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class MnstBagsGenerator:
    def __init__(
        self,
        embedding_tensor_path,
        label_tensor_path,
        batch_size=64,
        bag_length_dist="normal",
        max_bag_length=30,
        mean_bag_length=20,
        target_number=9,
        target_multiples=1,
        var_bag_length=5,
        num_bag=250,
        seed=1,
    ):
        self.batch_size = batch_size
        self.target_number = target_number
        self.target_multiples = target_multiples
        self.bag_length_dist = bag_length_dist
        self.max_bag_length = max_bag_length
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.embedding = torch.load(embedding_tensor_path)  # MNST train size x embedding size
        self.labels = torch.load(label_tensor_path)
        self.embedding_size = self.embedding.shape[1]
        assert len(self.embedding) == len(
            self.labels
        ), "Number of embedding is different from number of labels, please double check your data"
        self.seed = seed

    def __len__(self):
        return self.num_bag

    def dataloader(self, random_seed=None, return_indices=False):
        if random_seed is not None:
            self.seed = random_seed
        np.random.seed(self.seed)
        for _ in range(self.num_bag):
            ## Generate B x n_seq random numbers
            batched_random_indices = np.random.randint(len(self.embedding), size=(self.batch_size, self.max_bag_length))
            if self.bag_length_dist == "normal":
                random_bag_lengths = np.clip(
                    np.random.normal(
                        self.mean_bag_length,
                        self.var_bag_length,
                        size=(self.batch_size),
                    ).astype(int),
                    1,
                    self.max_bag_length,
                )
            elif self.bag_length_dist == "poisson":
                random_bag_lengths = np.clip(
                    np.random.poisson(self.mean_bag_length, size=(self.batch_size)).astype(int),
                    1,
                    self.max_bag_length,
                )
            else:
                random_bag_lengths = np.clip(
                    np.random.poisson(self.mean_bag_length, size=(self.batch_size)).astype(int),
                    1,
                    self.max_bag_length,
                )
            attention_mask = torch.tensor(
                [[1] * l + [0] * (self.max_bag_length - l) for l in random_bag_lengths], dtype=(torch.float32)
            )
            input_tensor = self.embedding[batched_random_indices]
            label_tensor = (
                torch.sum((self.labels[batched_random_indices] == self.target_number) * attention_mask, axis=1)
                >= self.target_multiples
            ).to(torch.float32)
            if return_indices:
                yield input_tensor, label_tensor, attention_mask, torch.tensor(batched_random_indices)
            else:
                yield input_tensor, label_tensor, attention_mask


class NumpyDataset:
    def __init__(
        self, inp_csv, array_path_col="array_path", target_col="target", bag_length_col="bag_length", max_bag_length=40
    ):
        self.df = pd.read_csv(inp_csv).reset_index(drop=True)
        self.array_path_col = array_path_col
        self.target_col = target_col
        self.bag_length_col = bag_length_col
        self.max_bag_length = max_bag_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        try:
            np_path = self.df.loc[index, self.array_path_col]
            label = self.df.loc[index, self.target_col]
            np_array = np.load(np_path)
            bag_length = len(np_array)
            indices = np.arange(bag_length)
            np.random.shuffle(indices)
            np_array = np_array[indices]
            attention_mask = torch.ones((self.max_bag_length), dtype=(torch.float32))
            if bag_length < self.max_bag_length:
                np_array = np.pad(
                    np_array, ((0, self.max_bag_length - bag_length), (0, 0)), "constant", constant_values=(0, 0)
                )
                attention_mask[bag_length:] = 0
            else:
                np_array = np_array[: self.max_bag_length, :]
            return torch.tensor(np_array), torch.tensor(label, dtype=torch.float32), attention_mask
        except:
            return None, None, None


class NumpyGenerator:
    def __init__(self, dataset, *argv, **kwargs):
        self.loader = torch.utils.data.DataLoader(dataset, *argv, **kwargs)

    def __len__(self):
        return len(self.loader)

    def dataloader(self, random_seed=None):
        return self.loader



if __name__ == "__main__":

    train_loader = MnstBagsGenerator(
        embedding_tensor_path="datasets/mnst_train_dinov2_small.pt",
        label_tensor_path="datasets/mnst_train_labels.pt",
        target_number=9,
        max_bag_length=30,
        num_bag=250,
    )
    t0 = time()
    print("There are %i bags" % (len(train_loader)))
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader.dataloader()):
        assert len(label_) == len(inp_), "label and input tensor should have the same batch size"

    print("MnstBagsGenerator takes %0.2f secs to generate %i samples" % (time() - t0, 64 * 250))

    t1 = time()
    dataset = NumpyDataset(inp_csv="datasets/mnst_small.csv")
    train_loader2 = NumpyGenerator(
        dataset, batch_size=128, num_workers=4, pin_memory=True, shuffle=True, drop_last=True
    )
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader2.dataloader()):
        assert len(label_) == len(inp_), "label and input tensor should have the same batch size"
        # print(batch_i)
    t2 = time()
    print("NumpyGenerator takes %0.3f secs to generate %i samples" % (t2 - t1, len(dataset)))

    t3 = time()
    train_loader3 = NumpyConcurrentGenerator("datasets/mnst_small.csv", batch_size=1024 * 4, max_threads=1024 * 2)
    for batch_i, (inp_, label_, mask_) in enumerate(train_loader3.dataloader()):
        assert len(label_) == len(inp_), "label and input tensor should have the same batch size"
        # print(batch_i)
    t4 = time()
    print("NumpyConcurrentGenerator takes %0.3f secs to generate %i samples" % (t4 - t3, len(train_loader3)))
