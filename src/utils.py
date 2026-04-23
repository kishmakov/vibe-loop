import torch

from dataclasses import asdict

def save_checkpoint(model, optimizer, cfg, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
    }, path)