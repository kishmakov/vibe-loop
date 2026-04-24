import aim
import math
import torch

from dataclasses import dataclass

from src.model.base_model import BaseModel
from src.utils import save_checkpoint


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 239
    report_interval: int = 10


class Training:
    def __init__(self, config: TrainingConfig, model: BaseModel):
        self.config = config
        self.model = model

        # Initialize a new run
        self.run = aim.Run(repo='/tmp/trn')
        self.run["model"] = self.model.config.__dict__
        self.run["training"] = self.config.__dict__

        model.init(config.seed)

        self.X_train, self.Y_train = self.model.get_train_batch()
        self.X_test, self.Y_test = self.model.get_test_batch()

        print("X_train.shape:", tuple(self.X_train.shape))
        print("Y_train.shape:", tuple(self.Y_train.shape))

    def report(self, epoch, loss):
        if epoch % self.config.report_interval != self.config.report_interval - 1:
            return

        self.run.track(loss.item(), name='loss', step=epoch, context={"subset":"train"})
        self.run.track(math.log(loss.item()), name='log_loss', step=epoch, context={"subset":"train"})

        with torch.no_grad():
            out = self.model.get_test_metrics(self.X_test, self.Y_test)

            for (metric_name, metric_value) in out.items():
                self.run.track(metric_value, name=metric_name, step=epoch, context={"subset":"test"})

    def train(self) -> None:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(self.X_train), self.Y_train)
            loss.backward()
            optimizer.step()
            self.report(epoch, loss)

        save_checkpoint(self.model, optimizer, self.model.config, self.config.epochs, "data/model.pt")
        self.run.close()






