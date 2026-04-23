import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class BitsToNumsConfig:
    bits: int = 4
    nums: int = 1 << bits  # 16
    seed: int = 239
    hidden: int = 10
    epochs: int = 50000
    lr: float = 1e-3


class BitsToNumsNet(nn.Module):
    def __init__(self, cfg: BitsToNumsConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.bits, cfg.hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(cfg.hidden, cfg.nums)

        torch.manual_seed(cfg.seed)

        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(layer.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_bits_vector(id: int, bits: int) -> torch.Tensor:
    vec = torch.zeros(bits)
    for b in range(bits):
        vec[bits - 1 - b] = (id >> b) & 1

    return vec

def get_nums_vector(id: int, nums: int) -> torch.Tensor:
    vec = torch.zeros(nums)
    vec[id] = 1
    return vec