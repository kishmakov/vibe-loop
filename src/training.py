import aim
import torch

from dataclasses import dataclass

from src.model.base_model import BaseModel
from src.utils import save_checkpoint


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 239

class Training:
    def __init__(self, config: TrainingConfig, model: BaseModel):
        self.config = config
        self.model = model
        model.init(config.seed)

    def train(self) -> None:
        X, Y = self.model.prepare_batch()
        print("Input shape:", tuple(X.shape))
        print("Output shape:", tuple(Y.shape))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # Initialize a new run
        run = aim.Run(repo='/tmp/trn')
        run["model"] = self.model.config.__dict__
        run["training"] = self.config.__dict__

        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(X), Y)
            loss.backward()
            optimizer.step()

            if epoch% 10 == 9:
                with torch.no_grad():
                    out = self.model.get_predictions(X)
                    rmse = torch.sqrt(torch.mean((out["preds"] - Y) ** 2))
                    acc = (out["pred_idx"] == Y.argmax(dim=-1)).float().mean()

                run.track(loss.item(), name='loss', step=epoch, context={"subset":"train"})
                run.track(rmse, name='rmse', step=epoch, context={"subset":"train"})
                run.track(acc, name='acc', step=epoch, context={"subset":"train"})

        save_checkpoint(self.model, optimizer, self.model.config, self.config.epochs, "data/model.pt")






