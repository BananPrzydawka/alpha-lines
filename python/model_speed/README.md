# Model inference speed

From the repository root, run all three configured models on Modal:

```sh
PYTHONPATH=python .venv/bin/modal run python/model_speed/modal_speed.py
```

Or on a local CUDA GPU:

```sh
PYTHONPATH=python .venv/bin/python -m model_speed.benchmark
```

Defaults: batch 4096, 8 warmup calls (including compilation), then 32 timed
forward calls. Override with `--batch`, `--calls`, `--warmup`, or select one
model with `--model resnet|maia|katago`.

Uses evaluation/inference mode, BF16 weights and inputs, channels-last layout,
cuDNN benchmarking, and `torch.compile(mode="max-autotune", fullgraph=True,
dynamic=False)` with its CUDA graph optimization. Models use fresh random
weights and the current `config.json` architecture settings.

Reports average GPU milliseconds per full batch using CUDA events, synchronized
wall milliseconds per call, and boards/second. Compilation and warmup are
reported separately. The same synthetic board batch stays on the GPU throughout;
data generation, encoding, transfers, and output checks are outside timing.
This measures inference, without backward passes or optimizer updates.
Max autotuning can take much longer than the 32-call measurement itself.
Results and configuration are printed as JSON; no checkpoints or volume commits.
