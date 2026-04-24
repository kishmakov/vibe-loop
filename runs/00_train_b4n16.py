import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.bits_to_nums import BitsToNumsNet, BitsToNumsConfig
from src.training import TrainingConfig, Training


if __name__ == "__main__":
    model_config = BitsToNumsConfig(bits=4)
    training_config = TrainingConfig(epochs=60000, device="cpu")

    model = BitsToNumsNet(model_config)
    training = Training(training_config, model)

    training.train()

