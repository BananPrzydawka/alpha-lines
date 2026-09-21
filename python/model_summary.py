"""Print a model's layers, parameters, operations, and estimated memory usage."""
import argparse

import torch
from torchinfo import summary

from config import height, width
from models.katago import KataGoNet
from models.maia import MaiaNet
from models.resnet import ResNet

MODELS = {"resnet": ResNet, "maia": MaiaNet, "katago": KataGoNet}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    if args.batch < 1 or args.depth < 1:
        parser.error("--batch and --depth must be positive")

    net = MODELS[args.model]().to(dtype=torch.bfloat16)
    inputs = [torch.zeros(args.batch, 5, height, width, dtype=torch.bfloat16)]
    if args.model != "maia":
        inputs.append(torch.zeros(args.batch, 2, dtype=torch.int64))
    stats = summary(
        net,
        input_data=inputs,
        device="cpu",
        mode="eval",
        depth=args.depth,
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        row_settings=("var_names",),
        verbose=0,
    )
    # Include parameters used through functional calls, such as Maia's templates.
    stats.total_param_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    print(stats)
    print("\nBF16 estimates; memory and multiply-add totals may omit functional "
          "operations (including attention). These are not peak memory measurements.")


if __name__ == "__main__":
    main()
