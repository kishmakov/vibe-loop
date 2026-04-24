from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Interface every trainable model must implement."""

    @abstractmethod
    def init(self, seed: int) -> None:
        """Set weights deterministically."""
        ...

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        ...

    @abstractmethod
    def prepare_batch(self):
        """Prepare a batch of data."""
        ...