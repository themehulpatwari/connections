# Ask an LLM to reason step-by-step before committing to a grouping.
# Run:  python experiments/llm_cot.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from connections import evaluate, load_puzzles


def solve(words):
    # TODO: prompt the LLM to think through candidates, then parse its 4 groups.
    raise NotImplementedError


if __name__ == "__main__":
    evaluate("llm_cot", solve, load_puzzles())
