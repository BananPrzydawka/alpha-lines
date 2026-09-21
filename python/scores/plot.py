# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9,<4"]
# ///
"""Plot all saved score runs: uv run --script python/scores/plot.py."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def load_runs(root, x_key):
    # Prefer JSON when both files exist: they describe the same run.
    directories = sorted({p.parent for name in ("results.json", "metrics.csv")
                          for p in root.rglob(name)})
    runs = []
    for directory in directories:
        try:
            path = directory / "results.json"
            if path.exists():
                rows = json.loads(path.read_text())["metrics"]
            else:
                with (directory / "metrics.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
            points = sorted((float(r[x_key]), float(r["val_loss"])) for r in rows)
            if not points or any(not math.isfinite(x) or not math.isfinite(y)
                                 or x < 0 or y < 0 for x, y in points):
                raise ValueError("missing or invalid metric points")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Skipping {directory}: {exc}", file=sys.stderr)
            continue
        label = str(directory.relative_to(root)) if directory != root else root.name
        runs.append((label, points))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("logs/score-symmetry"))
    parser.add_argument("--output", type=Path, help="Output image (default: ROOT/validation-loss.png)")
    parser.add_argument("--log-y", action="store_true", help="Logarithmic loss axis")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--time", dest="x_axis", action="store_const", const="elapsed_s",
                      help="Plot val loss over elapsed seconds (default)")
    mode.add_argument("--steps", dest="x_axis", action="store_const", const="step",
                      help="Plot val loss over training steps")
    parser.set_defaults(x_axis="elapsed_s")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_runs(args.root, args.x_axis)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for label, points in runs:
        x, y = zip(*points)
        ax.plot(x, y, label=label, linewidth=1.6, marker="." if len(points) == 1 else None)
    xlabel = "Training step" if args.x_axis == "step" else "Elapsed time (seconds)"
    ax.set(title="Score prediction: validation loss", xlabel=xlabel,
           ylabel="Validation cross-entropy")
    ax.grid(True, alpha=0.25)
    if args.log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(ax.get_ylim()[0], 0.1), top=4.5)
    else:
        ax.set_ylim(0, 4.5)
    if runs:
        ax.legend(title="Run", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="small")
    else:
        ax.text(.5, .5, f"No saved runs yet\n{args.root}", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    default_name = "validation-loss-steps.png" if args.x_axis == "step" else "validation-loss.png"
    output = args.output or args.root / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
