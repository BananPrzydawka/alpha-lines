"""Export the random-init BF16 model for native Rust MCTS inference."""
import argparse
from pathlib import Path

import torch
from config import height, width, benchmark
from model import alpha_lines_net


class MCTSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = alpha_lines_net().eval().to(torch.bfloat16)
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
        return self.from_planes(self.encode(cells, scores))

    def from_planes(self, planes):
        policy, value, _, _ = self.model(planes)
        priors = policy.flatten(1).float().softmax(1)[:, self.indices].contiguous()
        wdl = value.float().softmax(1)
        return priors, (wdl[:, 0] - wdl[:, 2]).contiguous()


class EncodedMCTSModel(MCTSModel):
    def forward(self, planes):
        return self.from_planes(planes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, help="Rust Config.b (model processes 2B perspectives)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=benchmark["seed"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default=benchmark["device"], help="CPU is for smoke checks only")
    parser.add_argument("--encoded-input", action="store_true", help="Accept BF16 [2B,7,10,16] planes; omit board encoding")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.output.suffix != ".pt2":
        parser.error("--output must end in .pt2")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; export on the GPU used for the benchmark")
    if (height, width) != (10, 16):
        parser.error("Rust engine requires a 10 x 16 board")
    torch.manual_seed(args.seed)
    model = (EncodedMCTSModel() if args.encoded_input else MCTSModel()).to(args.device).eval()
    inputs = (torch.zeros(args.batch, 80, dtype=torch.uint8, device=args.device),
              torch.zeros(args.batch, 2, dtype=torch.int32, device=args.device))
    if args.encoded_input:
        inputs = (torch.zeros(2 * args.batch, 7, height, width, dtype=torch.bfloat16,
                              device=args.device).contiguous(memory_format=torch.channels_last),)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(args.output),
            inductor_configs={"max_autotune": True, "coordinate_descent_tuning": True},
        )
    schema = "alpha-lines-encoded-v1" if args.encoded_input else "alpha-lines-mcts-v1"
    Path(str(args.output) + ".meta").write_text(f"{schema} {args.batch} {args.device}\n")
    print(f"Exported BF16 max-autotune model: B={args.batch}, model batch={2 * args.batch}, {args.output}")


if __name__ == "__main__":
    main()
