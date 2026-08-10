# Baseline: shuffle the 16 words into 4 arbitrary groups. The floor everything
# else should beat.  Run:  python experiments/random.py

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from connections import evaluate, load_puzzles


def solve(words):
    words = list(words)
    random.shuffle(words)
    return [words[i:i + 4] for i in range(0, 16, 4)]


if __name__ == "__main__":
    random.seed(5460)
    evaluate("random_baseline", solve, load_puzzles())
