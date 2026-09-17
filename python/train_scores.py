"""Online supervised learning of current scores from uniformly random Rust games."""
import csv
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from config import settings
from model import alpha_lines_net
from score_data import RandomScoreBatch, ScoreEncoder


def format_metrics(row):
    train_loss = "      -" if row["train_loss"] is None else f'{row["train_loss"]:7.3f}'
    return (
        f'Step {row["step"]:7,d} | boards {row["boards"]:11,d} | '
        f'games {row["completed_games"]:8,d} | {row["elapsed_s"]:8.1f}s | '
        f'loss train {train_loss}  val {row["val_loss"]:7.3f}'
        f'             P0 acc {row["p0_accuracy"]:6.1%}  MAE {row["p0_mae"]:6.2f} | '
        f'P1 acc {row["p1_accuracy"]:6.1%}  MAE {row["p1_mae"]:6.2f} | '
        f'zero acc {row["zero_baseline_accuracy"]:6.1%}  MAE {row["zero_baseline_mae"]:6.2f}'
    )


def validate_options(batch, steps, lr, eval_every, validation_steps, seed, seconds=600):
    if min(batch, steps, eval_every, validation_steps) < 1:
        raise ValueError("batch, steps, eval_every and validation_steps must be positive")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be in [0, 2**63 - 1)")


def train(library, output, *, batch=256, steps=1000, lr=0.001,
          eval_every=100, validation_steps=64, seed=1, device="cuda", on_save=None, seconds=600):
    validate_options(batch, steps, lr, eval_every, validation_steps, seed, seconds)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    encoder = ScoreEncoder().to(device)
    net = alpha_lines_net(score_prediction=True).to(device=device, memory_format=torch.channels_last)
    # Keep the original module for portable checkpoint keys. Compilation is lazy:
    # evaluation and training specialize separately, including the backward graph.
    forward_net = net
    if device == "cuda":
        print("Compiling model with torch.compile(mode='default'); first evaluation and training steps include compilation.", flush=True)
        forward_net = torch.compile(net, mode="default", fullgraph=True, dynamic=False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    use_amp = device == "cuda"
    # Fixed held-out trajectories from a separate random stream. Include early,
    # middle and terminal boards; never feed these positions to the optimizer.
    validation = []
    with RandomScoreBatch(library, batch, seed + 1) as source:
        for _ in range(validation_steps):
            cells, targets = source.step()
            validation.append((cells.clone(), targets.clone()))

    @torch.inference_mode()
    def evaluate():
        net.eval()
        totals = torch.zeros(7, device=device)
        for cells, targets in validation:
            targets = targets.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_net(encoder(cells.to(device)))
            predicted = logits.argmax(-1)
            totals[0] += F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="sum")
            totals[1:3] += (predicted == targets).sum(0)
            totals[3:5] += (predicted - targets).abs().sum(0)
            totals[5] += (targets == 0).sum()
            totals[6] += targets.sum()
        values = (totals / (batch * validation_steps)).cpu().tolist()
        net.train()
        return dict(val_loss=values[0] / 2, p0_accuracy=values[1], p1_accuracy=values[2],
                    p0_mae=values[3], p1_mae=values[4],
                    zero_baseline_accuracy=values[5] / 2, zero_baseline_mae=values[6] / 2)

    options = dict(batch=batch, steps=steps, lr=lr, eval_every=eval_every,
                   validation_steps=validation_steps, seed=seed, device=device, seconds=seconds)
    rows = []
    started = time.monotonic()
    interval_loss = torch.zeros((), device=device)
    interval_steps = 0

    def save(step, completed):
        nonlocal interval_steps
        metrics = evaluate()
        row = dict(step=step, boards=step * batch, completed_games=completed,
                   elapsed_s=time.monotonic() - started,
                   train_loss=interval_loss.item() / interval_steps if interval_steps else None,
                   **metrics)
        rows.append(row)
        print(format_metrics(row), flush=True)
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        result = dict(options=options, config=settings, torch=str(torch.__version__), metrics=rows)
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                        step=step, options=options, config=settings,
                        input_planes=["unplayable", "empty", "removed", "player_0", "player_1"],
                        output_players=[0, 1]), output / "checkpoint.pt")
        if on_save:
            on_save()
        interval_loss.zero_()
        interval_steps = 0
        return result

    result = save(0, 0)
    step = 0
    with RandomScoreBatch(library, batch, seed) as source:
        for next_step in range(1, steps + 1):
            if time.monotonic() - started >= seconds:
                break
            step = next_step
            cells, targets = source.step()
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_net(encoder(cells.to(device)))
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            interval_loss += loss.detach()
            interval_steps += 1
            # Drain GPU work so the deadline measures completed updates.
            if device == "cuda":
                torch.cuda.synchronize()
            expired = time.monotonic() - started >= seconds
            if step % eval_every == 0 or step == steps or expired:
                result = save(step, source.completed)
            if expired:
                break
        if rows[-1]["step"] != step:
            result = save(step, source.completed)
    return result
