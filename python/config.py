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

for name in ("katago_model", "katago_tf_model", "resnet_model"):
    options = settings[name]
    for key in ("filters", "blocks", "se_hidden",
                "policy_filters", "opponent_policy_filters", "action_value_filters", "mark_class_filters", "immediate_score_filters",
                "discounted_score_filters", "discounted_mark_filters"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{name}.{key} must be a positive integer")
    norm = options["group_norm"]
    if type(norm["groups"]) is not int or norm["groups"] < 1:
        raise ValueError(f"{name}.group_norm.groups must be a positive integer")
    if not math.isfinite(norm["eps"]) or norm["eps"] <= 0:
        raise ValueError(f"{name}.group_norm.eps must be positive and finite")
    if type(norm["affine"]) is not bool:
        raise ValueError(f"{name}.group_norm.affine must be a boolean")

tf = settings["katago_tf_model"]
for key in ("attention_heads", "mlp_ratio"):
    if type(tf[key]) is not int or tf[key] < 1:
        raise ValueError(f"katago_tf_model.{key} must be a positive integer")
if (tf["filters"] // 2) % tf["attention_heads"]:
    raise ValueError("katago_tf_model.attention_heads must divide half the filters")

maia = settings["maia_model"]
if type(maia["maia_big_version"]) is not bool:
    raise ValueError("maia_model.maia_big_version must be a boolean")
for key in ("dim", "layers", "head_dim", "mlp_ratio", "gab_dim"):
    if type(maia[key]) is not int or maia[key] < 1:
        raise ValueError(f"maia_model.{key} must be a positive integer")
if maia["dim"] % maia["head_dim"]:
    raise ValueError("maia_model.head_dim must divide dim")
for key in ("square_features", "hidden_dim"):
    if type(maia["score_head"][key]) is not int or maia["score_head"][key] < 1:
        raise ValueError(f"maia_model.score_head.{key} must be a positive integer")
