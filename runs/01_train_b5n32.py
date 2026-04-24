import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.bits_to_nums import BitsToNumsNet, BitsToNumsConfig
from src.training import TrainingConfig, Training
from src.utils import load_model


def make_b4_hook():
    b4_model = load_model(ROOT / "data" / "B4N16H2_cpu.pt", BitsToNumsNet, BitsToNumsConfig)
    b4_weight = b4_model.fc1.weight.detach()

    def hook(model):
        w = model.fc1.weight[:, :4].detach()
        rmse = torch.sqrt(torch.mean((w - b4_weight.to(w.device)) ** 2)).item()
        yield "fc1_b4_rmse", rmse

    return hook


if __name__ == "__main__":
    model_config = BitsToNumsConfig(bits=5)
    training_config = TrainingConfig(epochs=600000, device="cpu")

    model = BitsToNumsNet(model_config)
    training = Training(training_config, model)

    training.train(hook=make_b4_hook())

