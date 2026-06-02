import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import main

RUN_MODE = "current"
REWARD_MODE = "quantity_first"
GATE_TYPE = "book"

if __name__ == "__main__":
    main(RUN_MODE, REWARD_MODE, GATE_TYPE)
