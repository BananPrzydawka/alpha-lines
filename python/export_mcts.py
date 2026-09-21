"""Export the random-init BF16 model for native Rust MCTS inference."""
import argparse
from pathlib import Path

import torch
from config import height, width
from models.resnet import ResNet


class MCTSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError(
            "MCTS export is inactive: Rust expects scalar state values, but ResNet now has "
            "spatial action values. Update the native inference contract before enabling export."
        )
        self.model = ResNet().eval().to(torch.bfloat16)
        self.model.to(memory_format=torch.channels_last)
        squares = torch.arange(80)
        self.register_buffer("indices", squares // 8 * 16 + 2 * (squares % 8) + (squares // 8 % 2))
        self.register_buffer("marks", torch.tensor([1, 3, 4, 2], dtype=torch.uint8))
        self.register_buffer("left", torch.arange(width).view(1, 1, width) < width // 2)

    def encode(self, cells, scores):
        b = cells.shape[0]
        board = torch.zeros((b, height * width), dtype=torch.uint8, device=cells.device)
        board = board.scatter(1, self.indices.expand(b, -1), self.marks[cells.long()])
        board = board.view(b, height, width)
        planes = [(board == v).to(torch.bfloat16) for v in range(5)]
        opening = (cells == 0).all(1).view(b, 1, 1)
        own_score = (scores[:, 0].float() / 80).to(torch.bfloat16).view(b, 1, 1).expand(-1, height, width)
        opp_score = (scores[:, 1].float() / 80).to(torch.bfloat16).view(b, 1, 1).expand(-1, height, width)
        p0 = torch.stack([planes[0], planes[1] * (~opening | self.left),
                          planes[2], planes[3], planes[4], own_score, opp_score], 1)
        p1 = torch.stack([planes[0], planes[1] * (~opening | ~self.left),
                          planes[2], planes[4], planes[3], opp_score, own_score], 1)
        return torch.cat([p0, p1]).contiguous(memory_format=torch.channels_last)

    def forward(self, cells, scores):
        policy, value, _, _ = self.model(self.encode(cells, scores))
        priors = policy.flatten(1).float().softmax(1)[:, self.indices].contiguous()
        wdl = value.float().softmax(1)
        return priors, (wdl[:, 0] - wdl[:, 2]).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, help="Rust Config.b (model processes 2B perspectives)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="CPU is for smoke checks only")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.output.suffix != ".pt2":
        parser.error("--output must end in .pt2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; export on the GPU used for inference")
    if (height, width) != (10, 16):
        parser.error("Rust engine requires a 10 x 16 board")
    torch.manual_seed(args.seed)
    model = MCTSModel().to(args.device).eval()
    inputs = (torch.zeros(args.batch, 80, dtype=torch.uint8, device=args.device),
              torch.zeros(args.batch, 2, dtype=torch.int32, device=args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(args.output),
            inductor_configs={"max_autotune": True, "coordinate_descent_tuning": True},
        )
    Path(str(args.output) + ".meta").write_text(f"alpha-lines-mcts-v1 {args.batch} {args.device}\n")
    print(f"Exported BF16 max-autotune model: B={args.batch}, model batch={2 * args.batch}, {args.output}")


if __name__ == "__main__":
    main()
