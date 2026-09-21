"""Train the current-board score demo on the configured Modal GPU."""
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import uuid

import modal
from config import resources, settings
from modal_image import image, volume, project_dir

training = settings["score_training"]
score_image = image.add_local_dir(
    project_dir / "rust", remote_path="/root/rust",
    ignore=["target/", "tests/", "__pycache__/"],
)
app = modal.App("alphalines-score-demo")


@app.function(image=score_image, volumes={"/outputs": volume}, gpu=resources["gpu"],
              timeout=resources["timeout"])
def run_training(options: dict):
    from scores.train import train

    cache = Path("/outputs/score-demo-build")
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_HOME=str(cache / "cargo"), RUSTUP_HOME=str(cache / "rustup"),
               CARGO_TARGET_DIR=str(cache / "target"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cargo = Path(env["CARGO_HOME"]) / "bin/cargo"
    if not cargo.exists():
        installer = Path("/tmp/score-demo-rustup.sh")
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        subprocess.run(["sh", str(installer), "-y", "--profile", "minimal", "--no-modify-path"],
                       env=env, check=True)
    env["PATH"] = f"{cargo.parent}:{env['PATH']}"
    subprocess.run([str(cargo), "build", "--manifest-path", "/root/rust/Cargo.toml",
                    "--release", "--lib"], env=env, check=True)
    library = Path(env["CARGO_TARGET_DIR"]) / "release/libalpha_lines_game.so"
    output = Path("/outputs/score-demo") / uuid.uuid4().hex[:12]
    print(f"Saving metrics and checkpoint to {output}", flush=True)
    result = train(library, output, **options, on_save=volume.commit)
    result["remote_output"] = str(output)
    return result


@app.local_entrypoint()
def main(batch: int = training["batch"], steps: int = training["steps"],
         lr: float = training["lr"], eval_every: int = training["eval_every"],
         validation_steps: int = training["validation_steps"], seed: int = training["seed"],
         output: str = training["output"], seconds: float = training["seconds"],
         validation_batch: int = training["validation_batch"], model: str = "maia"):
    from scores.train import validate_options, validate_model

    validate_model(model)

    validate_options(batch, steps, lr, eval_every, validation_steps, seed, seconds, validation_batch)
    result = run_training.remote(dict(batch=batch, steps=steps, lr=lr, eval_every=eval_every,
                                      validation_steps=validation_steps, seed=seed, seconds=seconds,
                                      validation_batch=validation_batch, model=model))
    path = Path(output) / Path(result["remote_output"]).name
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    remote_dir = Path(result["remote_output"]).relative_to("/outputs")
    for name in ("checkpoint.pt", "metrics.csv"):
        temporary = path / (name + ".part")
        with temporary.open("wb") as stream:
            for chunk in volume.read_file(str(remote_dir / name)):
                stream.write(chunk)
        temporary.replace(path / name)
    print(f"Downloaded final checkpoint: {path / 'checkpoint.pt'}")
    print(f"Results: {path / 'results.json'}")
    print(f"Checkpoint and CSV: Modal volume alphalines, {result['remote_output']}")
