import aim
import pathlib
import sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.bits_to_nums import BitsToNumsNet, BitsToNumsConfig, get_bits_vector, get_nums_vector
from src.training import TrainingConfig
from src.utils import save_checkpoint


def train() -> None:
    model_config = BitsToNumsConfig()
    training_config = TrainingConfig()

    X = torch.stack([get_bits_vector(i, model_config.bits) for i in range(model_config.nums)])
    Y = torch.stack([get_nums_vector(i, model_config.nums) for i in range(model_config.nums)])

    print("Input shape:", tuple(X.shape))
    print("Output shape:", tuple(Y.shape))

    model = BitsToNumsNet(model_config)
    model.init(training_config.seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.lr)

    # Initialize a new run
    run = aim.Run(repo='/tmp/trn')
    run["cfg"] = model_config.__dict__

    for epoch in range(training_config.epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()

        if epoch% 10 == 9:
            with torch.no_grad():
                out = model.get_predictions(X)
                rmse = torch.sqrt(torch.mean((out["preds"] - Y) ** 2))
                acc = (out["pred_idx"] == Y.argmax(dim=-1)).float().mean()

            run.track(loss.item(), name='loss', step=epoch, context={"subset":"train"})
            run.track(rmse, name='rmse', step=epoch, context={"subset":"train"})
            run.track(acc, name='acc', step=epoch, context={"subset":"train"})

    save_checkpoint(model, optimizer, model_config, training_config.epochs, "data/model.pt")


if __name__ == "__main__":
    train()

