# Shared helpers: load the puzzles and score a solver's grouping.
#
# A puzzle is just a dict:
#   {"game_id": int, "date": "YYYY-MM-DD", "words": [16 words], "answer": [[4], [4], [4], [4]]}
# The four answer groups are ordered by difficulty, so answer[i] is the COLORS[i] group.
# A solver takes the 16 words and returns its own [[4], [4], [4], [4]] (order doesn't matter).

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from miss_distance import miss_distance, miss_distance_distribution

DATA_FILE = Path(__file__).parent / "data" / "Connections_Data.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# NYT group difficulty, easiest to hardest — matches Group Level 0..3 in the CSV.
COLORS = ("yellow", "green", "blue", "purple")


def load_puzzles():
    """Return every puzzle as a dict, oldest first. Run data.py first if the CSV is missing."""
    # "NA" is a real answer word (sodium), so don't let pandas read it as missing.
    df = pd.read_csv(DATA_FILE, keep_default_na=False, na_values=[])

    puzzles = []
    for (date, game_id), rows in df.groupby(["Puzzle Date", "Game ID"], sort=True):
        rows = rows.sort_values(["Starting Row", "Starting Column"])
        answer = [
            list(group["Word"])
            for _, group in rows.groupby("Group Level", sort=True)  # 0..3 -> COLORS order
        ]
        puzzles.append({
            "game_id": int(game_id),
            "date": date,
            "words": list(rows["Word"]),
            "answer": answer,
        })
    return puzzles


def solved_colors(guess, answer):
    """Which colors the guess got exactly right, as a set of color names.

    answer[i] is the COLORS[i] group; a color counts as solved if any guessed
    group matches its four words exactly.
    """
    guessed = {frozenset(group) for group in guess}
    return {
        COLORS[i]
        for i, group in enumerate(answer)
        if frozenset(group) in guessed
    }


def evaluate(name, solve, puzzles):
    """Run `solve(words) -> 4 groups` over every puzzle, print the metrics, and
    save a summary + per-puzzle breakdown to results/{name}-{timestamp}.json.

    Reports grouping accuracy (groups solved overall), per-color accuracy, and
    win rate (puzzles where all 4 groups were solved).
    """
    color_hits = {color: 0 for color in COLORS}
    wins = 0
    per_puzzle = []
    per_puzzle_missed = []
    for puzzle in puzzles:
        guess = solve(puzzle["words"])
        solved = solved_colors(guess, puzzle["answer"])
        for color in solved:
            color_hits[color] += 1
        if len(solved) == 4:
            wins += 1
        missed = miss_distance(guess, puzzle["answer"], COLORS)
        per_puzzle_missed.append(missed)
        per_puzzle.append({
            "game_id": puzzle["game_id"],
            "date": puzzle["date"],
            "solved_colors": sorted(solved, key=COLORS.index),
            "miss_distance": missed,
        })

    n = len(puzzles)
    total_hits = sum(color_hits.values())
    grouping_accuracy = total_hits / (n * 4)
    miss_dist = miss_distance_distribution(per_puzzle_missed, COLORS)

    print(f"puzzles:           {n}")
    print(f"grouping accuracy: {grouping_accuracy:.3f}  ({total_hits}/{n * 4} groups)")
    print("color accuracy:")
    for color in COLORS:
        print(f"  {color:<8} {color_hits[color] / n:.3f}")
    print(f"win rate:          {wins / n:.3f}")
    print("miss distance (0 = solved exactly):")
    for color in COLORS:
        counts = miss_dist[color]
        print(
            f"  {color:<8} "
            + " ".join(f"{k}:{counts[k]}" for k in range(5))
        )

    summary = {
        "puzzles": n,
        "grouping_accuracy": grouping_accuracy,
        "color_accuracy": {color: color_hits[color] / n for color in COLORS},
        "win_rate": wins / n,
        "miss_distance_distribution": miss_dist,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = RESULTS_DIR / f"{name}-{stamp}.json"
    output_file.write_text(json.dumps(
        {"solver": name, "timestamp": stamp, "summary": summary, "puzzles": per_puzzle},
        indent=2,
    ))
    print(f"\nSaved to {output_file}")

    return summary
