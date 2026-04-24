import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from src.model.base_model import BaseModel


@dataclass
class BitsToNumsConfig:
    bits: int
    hidden: int = 2
    nums: int = field(init=False)

    def __post_init__(self):
        self.nums = 1 << self.bits


class BitsToNumsNet(BaseModel, nn.Module):
    def __init__(self, config: BitsToNumsConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.bits, config.hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden, config.nums)

    def init(self, seed):
        torch.manual_seed(seed)

        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def prepare_batch(self):
        X = torch.stack([_get_bits_vector(i, self.config.bits) for i in range(self.config.nums)])
        Y = torch.stack([_get_nums_vector(i, self.config.nums) for i in range(self.config.nums)])
        return X, Y

    @torch.no_grad()
    def get_predictions(self, x: torch.Tensor):
        logits = self.forward(x)

        probs = torch.softmax(logits, dim=-1)              # probabilities
        pred_idx = probs.argmax(dim=-1)                    # indices
        preds = F.one_hot(pred_idx, num_classes=probs.shape[-1]).float()

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "pred_idx": pred_idx,
        }


def _get_bits_vector(id: int, bits: int) -> torch.Tensor:
    vec = torch.zeros(bits)
    for b in range(bits):
        vec[bits - 1 - b] = (id >> b) & 1

    return vec

def _get_nums_vector(id: int, nums: int) -> torch.Tensor:
    vec = torch.zeros(nums)
    vec[id] = 1
    return vec

