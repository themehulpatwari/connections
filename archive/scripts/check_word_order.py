# One-off data-leak check: does the entry order stored in the dataset give away
# group membership? Any order-sensitive method (one forward pass over all 16
# entries, or an LLM prompt) reads that order, so if the stored order carries
# any signal about the grouping, the method is being handed part of the answer.
#
# The test is two-sided. Order grouped BY color would be the obvious leak, but
# order that avoids putting group members together leaks just as usefully — it
# tells an order-sensitive method which pairs are unlikely to belong together.
#
#   venv/bin/python scripts/check_word_order.py

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))  # repo root (this file lives in archive/scripts/)
from connections import COLORS, load_puzzles

SAMPLE = 10
SEED = 0

# Chance rates under a uniformly random entry order.
FIRST_FOUR_CHANCE = (3 / 15) * (2 / 14) * (1 / 13)  # all of entries 0-3 in one group
PAIR_CHANCE = 3 / 15                                # a given pair of entries in one group


def levels(puzzle):
    """Group index 0-3 (COLORS order) for each of the 16 entries, in stored order."""
    level = {word: i for i, group in enumerate(puzzle["answer"]) for word in group}
    return [level[word] for word in puzzle["words"]]


def rate(same, total, label):
    z = (same / total - PAIR_CHANCE) / (PAIR_CHANCE * (1 - PAIR_CHANCE) / total) ** 0.5
    print(f"  {label:<34} {same / total:.4f} ({same}/{total})   chance {PAIR_CHANCE:.4f}   z = {z:+.1f}")
    return z


def main():
    puzzles = load_puzzles()

    print(f"=== {SAMPLE} random puzzles: stored entry order vs. true group ===\n")
    for puzzle in random.Random(SEED).sample(puzzles, SAMPLE):
        print(f"game {puzzle['game_id']}  ({puzzle['date']})")
        for position, (word, level) in enumerate(zip(puzzle["words"], levels(puzzle))):
            print(f"  {position:>2}  {word:<18} {COLORS[level]}")
        print(f"  levels in stored order: {levels(puzzle)}\n")

    # Stored order is the puzzle's board reading order (load_puzzles sorts by
    # Starting Row, Starting Column), so position i is board cell (i // 4, i % 4)
    # and the two adjacencies below are the board's own rows and columns.
    first_four = 0
    counts = {"neighbours in stored order": [0, 0], "same row, side by side": [0, 0],
              "same column, one above": [0, 0]}
    for puzzle in puzzles:
        level = levels(puzzle)
        first_four += len(set(level[:4])) == 1
        for i in range(15):
            counts["neighbours in stored order"][0] += level[i] == level[i + 1]
            counts["neighbours in stored order"][1] += 1
        for r in range(4):
            for c in range(4):
                if c < 3:
                    counts["same row, side by side"][0] += level[4 * r + c] == level[4 * r + c + 1]
                    counts["same row, side by side"][1] += 1
                if r < 3:
                    counts["same column, one above"][0] += level[4 * r + c] == level[4 * r + 4 + c]
                    counts["same column, one above"][1] += 1

    print(f"=== full dataset ({len(puzzles)} puzzles) ===")
    print(f"  {'first four entries all one group':<34} {first_four / len(puzzles):.4f} "
          f"({first_four}/{len(puzzles)})   chance {FIRST_FOUR_CHANCE:.4f}")
    worst = max((abs(rate(same, total, label)) for label, (same, total) in counts.items()))

    print("\nVERDICT: " + (
        f"LEAK (|z| = {worst:.1f}). Stored entry order carries information about group "
        "membership. Shuffling before any order-sensitive method reads the entries is "
        "mandatory, not hygiene."
        if worst > 4 else
        f"NO LEAK (|z| = {worst:.1f}). Stored order is consistent with a random scramble. "
        "Shuffle anyway so order stays a controlled variable."
    ))


if __name__ == "__main__":
    main()
