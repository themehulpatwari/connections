# Ask an LLM for a direct grouping of the 16 words, no reasoning steps.
# Run:  python experiments/llm_zeroshot.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from connections import evaluate, load_puzzles


def solve(words):
    # TODO: prompt the LLM with the 16 words, parse its 4 groups of 4.
    raise NotImplementedError


if __name__ == "__main__":
    evaluate("llm_zeroshot", solve, load_puzzles())
