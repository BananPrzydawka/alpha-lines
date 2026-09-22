"""Load the shared project configuration; edit ../config.json to change defaults."""
import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
settings = json.loads(CONFIG_PATH.read_text())
game = settings["game"]
resources = settings["modal"]

if (game["height"], game["width"]) != (10, 16):
    raise ValueError("The Rust engine and model encoding require a 10 x 16 board")

height, width = game["height"], game["width"]
board_size = height * width

for name in ("resnet_model", "katago_model"):
    options = settings[name]
    for key in ("filters", "blocks", "se_hidden",
                "policy_filters", "action_value_filters", "immediate_score_filters"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{name}.{key} must be a positive integer")
    norm = options["group_norm"]
    if type(norm["groups"]) is not int or norm["groups"] < 1:
        raise ValueError(f"{name}.group_norm.groups must be a positive integer")
    if not math.isfinite(norm["eps"]) or norm["eps"] <= 0:
        raise ValueError(f"{name}.group_norm.eps must be positive and finite")
    if type(norm["affine"]) is not bool:
        raise ValueError(f"{name}.group_norm.affine must be a boolean")
