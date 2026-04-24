import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    """Interface every trainable model must implement."""

    @property
    @abstractmethod
    def config(self):
        """Model config."""
        ...

    @abstractmethod
    def init(self, seed: int) -> None:
        """Set weights deterministically."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...

    @abstractmethod
    def get_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of train data."""
        ...

    @abstractmethod
    def get_test_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare a batch of test data."""
        ...

    @abstractmethod
    def get_test_metrics(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Compute test metrics."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Compute short name of the model."""
        ...