import aim
import pathlib
import sys
import time
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bits_to_nums import BitsToNumsNet


def main() -> None:
    model = BitsToNumsNet(4, 16)

    x = torch.randn(2, 4)
    y = model(x)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("First output row:", y[0].tolist())

    # Initialize a new run
    run = aim.Run(repo='/tmp/trn')

    # Log run parameters
    run["hparams"] = {
        "learning_rate": 0.001,
        "batch_size": 32,
    }

    # Log metrics
    for i in range(100000):
        time.sleep(0.01)  # Simulate training time
        run.track(i, name='loss', step=i, context={ "subset":"train" })
        run.track(i, name='acc', step=i, context={ "subset":"train" })


if __name__ == "__main__":
    main()
