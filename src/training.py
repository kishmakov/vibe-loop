import aim
import math
import pathlib
import time
import torch

from dataclasses import dataclass

from src.model.base_model import BaseModel
from src.utils import save_checkpoint, load_model


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 239
    report_interval: int = 10
    save_interval: int = 10000
    device: str = "cuda"


class Training:
    def __init__(self, config: TrainingConfig, model: BaseModel, model_weights: pathlib.Path = None):
        self.config = config
        self.model = model
        self._train_start: float = 0.0
        self._solved = False
        self._dir = pathlib.Path('/tmp/trn')

        # Initialize a new run
        self.run = aim.Run(repo=self._dir)
        self.run.name = model.get_name() + " @ " + config.device
        self.run["model"] = self.model.config.__dict__
        self.run["training"] = self.config.__dict__

        if (model_weights is not None):
            self.model = load_model(model_weights, type(model), type(model.config))
        else:
            model.init(config.seed)

        self.model.to(config.device)

        self.X_train, self.Y_train = self.model.get_train_batch()
        self.X_train = self.X_train.to(config.device)
        self.Y_train = self.Y_train.to(config.device)

        self.X_test, self.Y_test = self.model.get_test_batch()
        self.X_test = self.X_test.to(config.device)
        self.Y_test = self.Y_test.to(config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        print("X_train.shape:", tuple(self.X_train.shape))
        print("Y_train.shape:", tuple(self.Y_train.shape))

    def report(self, epoch, loss, hook):
        if epoch % self.config.report_interval != self.config.report_interval - 1:
            return

        self.run.track(loss.item(), name='loss', step=epoch, context={"subset":"train"})
        self.run.track(math.log(loss.item()), name='log_loss', step=epoch, context={"subset":"train"})

        mean_epoch_time = (time.time() - self._train_start) / (epoch + 1)
        self.run.track(math.log(mean_epoch_time), name='log_mean_epoch_time', step=epoch)

        with torch.no_grad():
            out = self.model.get_test_metrics(self.X_test, self.Y_test)
            if not self._solved and out["acc"] > 0.999999:
                self._solved = True
                print(f"Solved on epoch={epoch}")

            for (metric_name, metric_value) in out.items():
                self.run.track(metric_value, name=metric_name, step=epoch, context={"subset":"test"})

        if hook is not None:
            for (metric_name, metric_value) in hook(self.model):
                self.run.track(metric_value, name=metric_name, step=epoch, context={"subset":"hook"})

    def save_checkpoint(self, epoch):
        last = epoch == self.config.epochs - 1
        regular = epoch % self.config.save_interval == self.config.save_interval - 1
        if last or regular:
            save_checkpoint(self._dir, self.model, self.config, epoch, self.optimizer)


    def train(self, loss_cls, hook = None) -> None:
        criterion = loss_cls()

        self._train_start = time.time()
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            loss = criterion(self.model(self.X_train), self.Y_train)
            loss.backward()
            self.optimizer.step()

            self.report(epoch, loss, hook)
            self.save_checkpoint(epoch)

        self.run.close()

