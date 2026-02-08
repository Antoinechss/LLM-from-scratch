# Elementary Character Based Tokenizer : Mapping characters to integers
from dataset import chars
from typing import List
import torch


st_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_st = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    return torch.tensor([st_to_int[c] for c in s])


def decode(l: List[int]) -> str:
    return "".join([int_to_st[i] for i in l])


