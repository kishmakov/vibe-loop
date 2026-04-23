from dataclasses import dataclass


@dataclass
class TrainingConfig:
    seed: int = 239
    epochs: int = 100000
    lr: float = 1e-3

# class Training:
#     def __init__(self, cfg: TrainingConfig, model):
#         self.cfg = cfg
#         model.init
#         torch.manual_seed(cfg.seed)

#     def train(self, model: TinyDecoderLM, a: torch.Tensor, b: torch.Tensor) -> TinyDecoderLM:



