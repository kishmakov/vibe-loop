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


class BitsToNumsNet(BaseModel):
    def __init__(self, config: BitsToNumsConfig):
        super().__init__()
        self._config = config
        self.fc1 = nn.Linear(config.bits, config.hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden, config.nums)

    @property
    def config(self) -> BitsToNumsConfig:
        return self._config

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

    def get_train_batch(self):
        X = torch.stack([_get_bits_vector(i, self.config.bits) for i in range(self.config.nums)])
        Y = torch.stack([_get_nums_vector(i, self.config.nums) for i in range(self.config.nums)])
        return X, Y

    def get_test_batch(self):
        return self.get_train_batch()

    @torch.no_grad()
    def get_test_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1) # probabilities
        pred_idx = probs.argmax(dim=-1) # indices
        preds = F.one_hot(pred_idx, num_classes=probs.shape[-1]).float()

        return {
            "rmse": torch.sqrt(torch.mean((preds - y) ** 2)).item(),
            "acc": (pred_idx == y.argmax(dim=-1)).float().mean().item(),
        }

    def get_name(self) -> str:
        return f"B{self.config.bits}N{self.config.nums}H{self.config.hidden}"

def _get_bits_vector(id: int, bits: int) -> torch.Tensor:
    vec = torch.zeros(bits)
    for b in range(bits):
        vec[bits - 1 - b] = (id >> b) & 1

    return vec

def _get_nums_vector(id: int, nums: int) -> torch.Tensor:
    vec = torch.zeros(nums)
    vec[id] = 1
    return vec

