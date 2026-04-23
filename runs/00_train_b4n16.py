import aim
import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bits_to_nums import BitsToNumsNet, BitsToNumsConfig, get_bits_vector, get_nums_vector
from src.utils import save_checkpoint


def train() -> None:
    cfg = BitsToNumsConfig()

    X = torch.stack([get_bits_vector(i, cfg.bits) for i in range(cfg.nums)])
    Y = torch.stack([get_nums_vector(i, cfg.nums) for i in range(cfg.nums)])

    print("Input shape:", tuple(X.shape))
    print("Output shape:", tuple(Y.shape))

    model = BitsToNumsNet(cfg)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)


    # Initialize a new run
    run = aim.Run(repo='/tmp/trn')

    # Log run parameters
    run["cfg"] = cfg.__dict__

    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

        run.track(loss.item(), name='loss', step=epoch, context={"subset":"train"})

    save_checkpoint(model, optimizer, cfg, cfg.epochs, "data/model.pt")


def main() -> None:
    train()

if __name__ == "__main__":
    main()
