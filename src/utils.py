import dataclasses
import pathlib
import torch

from dataclasses import asdict
from enum import StrEnum

from src.model.base_model import BaseModel


class KEYS(StrEnum):
    MODEL_STATE = "model_state_dict"
    OPTIMIZER_STATE = "optimizer_state_dict"
    MODEL_CONFIG = "model_config"
    TRAINING_CONFIG = "training_config"
    EPOCH = "epoch"


def save_checkpoint(path: pathlib.Path, model: BaseModel, training_config, epoch, optimizer):
    file_name = model.get_name() + ".pt"

    torch.save({
        KEYS.MODEL_STATE: model.state_dict(),
        KEYS.OPTIMIZER_STATE: optimizer.state_dict(),
        KEYS.MODEL_CONFIG: asdict(model.config),
        KEYS.TRAINING_CONFIG: asdict(training_config),
        KEYS.EPOCH: epoch,
    }, path / file_name)


def _load_model(path, model_cls, model_cfg_cls):
    checkpoint = torch.load(path, weights_only=False)

    init_fields = {field.name for field in dataclasses.fields(model_cfg_cls) if field.init}
    model_config = model_cfg_cls(**{k: v for k, v in checkpoint[KEYS.MODEL_CONFIG].items() if k in init_fields})
    model = model_cls(model_config)
    model.load_state_dict(checkpoint[KEYS.MODEL_STATE])

    return model, checkpoint


def load_checkpoint(path, model_cls, model_cfg_cls, training_config_cls, optimizer_cls):
    model, checkpoint = _load_model(path, model_cls, model_cfg_cls)

    training_config = training_config_cls(**checkpoint[KEYS.TRAINING_CONFIG])

    optimizer = optimizer_cls(model.parameters(), lr=training_config.lr)
    optimizer.load_state_dict(checkpoint[KEYS.OPTIMIZER_STATE])

    return model, optimizer, training_config, checkpoint[KEYS.EPOCH]


def load_model(path, model_cls, model_cfg_cls):
    return _load_model(path, model_cls, model_cfg_cls)[0]