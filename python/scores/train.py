"""Online supervised learning of current scores from uniformly random Rust games."""
import csv
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from config import settings
from models.maia import MaiaNet
from scores.data import RandomScoreBatch, ScoreEncoder, both_perspectives

MODEL_TYPES = {"maia": MaiaNet}


def validate_model(model):
    if model not in MODEL_TYPES:
        raise ValueError("Score prediction training supports only maia; ResNet and KataGo now output policy and action values.")


def format_metrics(row):
    train_loss = "      -" if row["train_loss"] is None else f'{row["train_loss"]:7.3f}'
    phase_times = []
    for label, field in (("cpu", "cpu_ms"), ("model", "model_ms"), ("log", "log_ms")):
        value = row[field]
        duration = "-" if value is None else f"{value:.2f}ms"
        phase_times.append(f"{label} {duration}")
    return (
        f'Step {row["step"]:7,d} | boards {row["boards"]:11,d} | '
        f'games {row["completed_games"]:8,d} | {row["elapsed_s"]:8.1f}s | '
        f'loss train {train_loss}  val {row["val_loss"]:7.3f}'
        f' | P0 acc {row["p0_accuracy"]:6.1%} | '
        f'P1 acc {row["p1_accuracy"]:6.1%} | '
        f'zero acc {row["zero_baseline_accuracy"]:6.1%} | '
        + "  ".join(phase_times)
    )


def validate_options(batch, steps, lr, eval_every, validation_steps, seed, seconds=settings["score_training"]["seconds"], validation_batch=256):
    if validation_batch < 1:
        raise ValueError("validation_batch must be positive")
    if min(batch, steps, eval_every, validation_steps) < 1:
        raise ValueError("batch, steps, eval_every and validation_steps must be positive")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not 0 <= seed < 2**63 - 1:
        raise ValueError("seed must be in [0, 2**63 - 1)")


def train(library, output, *, batch=256, steps=1000, lr=0.001,
          eval_every=100, validation_steps=64, seed=1, device="cuda",
          on_save=None, model="maia",
          seconds=settings["score_training"]["seconds"], validation_batch=256):
    validate_options(batch, steps, lr, eval_every, validation_steps, seed, seconds, validation_batch)
    validate_model(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    encoder = ScoreEncoder().to(device)
    model_class = MODEL_TYPES[model]
    net = model_class()
    net = net.to(device=device, memory_format=torch.channels_last)
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
    with RandomScoreBatch(library, validation_batch, seed + 1) as source:
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
        values = (totals / (validation_batch * validation_steps)).cpu().tolist()
        net.train()
        return dict(val_loss=values[0] / 2, p0_accuracy=values[1], p1_accuracy=values[2],
                    p0_mae=values[3], p1_mae=values[4],
                    zero_baseline_accuracy=values[5] / 2, zero_baseline_mae=values[6] / 2)

    options = dict(batch=batch, steps=steps, lr=lr, eval_every=eval_every,
                   validation_steps=validation_steps, seed=seed, device=device, seconds=seconds,
                   training_perspectives=2, model_batch=2 * batch, model=model,
                   normalization="layernorm" if model == "maia" else "groupnorm",
                   validation_batch=validation_batch, validation_positions=validation_batch * validation_steps)
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    print(f"Model: {model} ({parameter_count:,} parameters)", flush=True)
    print(f"Training: {batch} boards × 2 player perspectives = {2 * batch} model examples/update; "
          f"validation: {validation_batch * validation_steps:,} positions.", flush=True)
    rows = []
    started = time.monotonic()
    interval_loss = torch.zeros((), device=device)
    interval_steps = 0
    timing = dict(cpu_s=0.0, model_s=0.0, logging_s=0.0)

    def synchronize():
        # Wall-clock phase timings must include completed CUDA work, not enqueue time.
        if device == "cuda":
            torch.cuda.synchronize()

    synchronize()

    def report(step, completed):
        nonlocal interval_steps
        logging_started = time.monotonic()
        metrics = evaluate()
        train_loss = interval_loss.item() / interval_steps if interval_steps else None
        synchronize()
        snapshot = time.monotonic()
        timing["logging_s"] += snapshot - logging_started
        row = dict(step=step, boards=step * batch, examples=step * batch * 2,
                   completed_games=completed,
                   elapsed_s=snapshot - started,
                   train_loss=train_loss,
                   **timing, **metrics)
        # Total phase milliseconds since the previous report (since startup for
        # step zero). Retain cumulative seconds in saved data as well.
        previous = rows[-1] if rows else None
        for seconds_key, milliseconds_key in (
            ("cpu_s", "cpu_ms"),
            ("model_s", "model_ms"),
            ("logging_s", "log_ms"),
        ):
            row[milliseconds_key] = (
                1000 * (timing[seconds_key] - (previous[seconds_key] if previous else 0))
            )
        rows.append(row)
        print(format_metrics(row), flush=True)
        interval_loss.zero_()
        interval_steps = 0
        synchronize()
        # Printing this report is included in the next one.
        timing["logging_s"] += time.monotonic() - snapshot

    report(0, 0)
    step = 0
    with RandomScoreBatch(library, batch, seed) as source:
        for next_step in range(1, steps + 1):
            if time.monotonic() - started >= seconds:
                break
            step = next_step
            preparation_started = time.monotonic()
            cells, targets = source.step()
            targets = targets.to(device)
            inputs, targets = both_perspectives(encoder(cells.to(device)), targets)
            synchronize()
            model_started = time.monotonic()
            timing["cpu_s"] += model_started - preparation_started
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_net(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            interval_loss += loss.detach()
            interval_steps += 1
            synchronize()
            timing["model_s"] += time.monotonic() - model_started
            expired = time.monotonic() - started >= seconds
            if step % eval_every == 0 or step == steps or expired:
                report(step, source.completed)
            if expired:
                break
        if rows[-1]["step"] != step:
            report(step, source.completed)
    torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, options=options, config=settings,
                    input_planes=["unplayable", "empty", "removed", "player_0", "player_1"],
                    output_players=[0, 1]), output / "checkpoint.pt")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(options=options, config=settings, torch=str(torch.__version__), metrics=rows)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if on_save:
        on_save()
    return result
