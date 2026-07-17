# Cluster the words by embedding similarity into 4 groups of 4.
# Run:  python experiments/embedding.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from connections import evaluate, load_puzzles


def solve(words):
    # TODO: embed the 16 words, cluster into 4 groups of 4, return them.
    raise NotImplementedError


if __name__ == "__main__":
    evaluate("embedding", solve, load_puzzles())
