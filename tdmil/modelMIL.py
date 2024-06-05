import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## The MILAttention is inspired by https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
MASK_VALUE = -1000


class MILAttention(nn.Module):
    def __init__(self, embedding_size=384, middle_dim=128, attention_branches=1):
        super(MILAttention, self).__init__()
        self.M = embedding_size
        self.L = middle_dim
        self.ATTENTION_BRANCHES = attention_branches

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),  # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )
        self.classifier = nn.Linear(self.M * self.ATTENTION_BRANCHES, 1)

    def forward(
        self,
        x,
        attention_mask,
        return_logits=True,
        return_attention=False,
        return_feature=False,
    ):
        """
        apply attention to the input
        x: batch_size x n_seq x embedding_size
        n_seq is the max bag length

        attention_mask: batch_size x n_seq
        attention_mask == 0 if the bag has no data at the index
        attention_mask == 1 if the bag has data at the index
        When attention_mask == 0, the attention mechnism will not count the attention there
        e.g.
        11111000
        11100000
        """
        batch_size, n_seq, emb_size = x.shape
        assert emb_size == self.M, "The input embeding size doesn't match the model embeding size"
        x = x * attention_mask.reshape(batch_size, n_seq, 1).to(torch.float32)
        A = self.attention(x)  # B x n_seq x ATTENTION_BRANCHES
        A = torch.transpose(A, -2, -1)  # B x ATTENTION_BRANCHES x n_seq
        A = A + MASK_VALUE * (1 - attention_mask.reshape(batch_size, 1, n_seq))
        A = F.softmax(A, dim=-1)  # softmax over K : B x ATTENTION_BRANCHES x n_seq
        Z = torch.matmul(A, x)  # B x ATTENTION_BRANCHES x emb_size
        ## Flatten the last two dimension for the classifier
        Z2 = Z.view(batch_size, -1)
        results = {}
        if return_logits:
            logits = self.classifier(Z2)
            results["logits"] = logits
        if return_attention:
            results["attention"] = A
        if return_feature:
            results["feature"] = Z
        return results


if __name__ == "__main__":

    model = MILAttention()
    inputs = torch.randn((16, 50, 384), dtype=(torch.float32))
    random_bag_lengths = np.clip(np.random.normal(10, 2, size=(16)).astype(int), 1, 50)
    attention_mask = torch.zeros((16, 50), dtype=(torch.float32))
    for i_, l_ in enumerate(random_bag_lengths):
        attention_mask[i_, :l_] = 1
    model_output = model(inputs, attention_mask, return_attention=True, return_feature=True)
