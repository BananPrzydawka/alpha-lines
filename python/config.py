"""Load the shared project configuration; edit ../config.json to change defaults."""
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
settings = json.loads(CONFIG_PATH.read_text())
game = settings["game"]
model = settings["model"]
mcts = settings["mcts"]
benchmark = settings["benchmark"]
resources = settings["modal"]

if (game["height"], game["width"]) != (10, 16):
    raise ValueError("The Rust engine and model encoding require a 10 x 16 board")

height, width = game["height"], game["width"]
board_size = height * width
