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


def _grouping_and_win(solve, puzzles):
    """Grouping accuracy (groups solved / 4n) and win rate (all-4 puzzles) for a
    puzzle subset. Empty subset -> nan metrics so callers can print a placeholder."""
    if not puzzles:
        return {"puzzles": 0, "grouping_accuracy": float("nan"), "win_rate": float("nan")}
    groups_solved = 0
    wins = 0
    for puzzle in puzzles:
        solved = solved_colors(solve(puzzle["words"]), puzzle["answer"])
        groups_solved += len(solved)
        if len(solved) == 4:
            wins += 1
    n = len(puzzles)
    return {
        "puzzles": n,
        "grouping_accuracy": groups_solved / (n * 4),
        "win_rate": wins / n,
    }


def evaluate_by_cutoff(solve, puzzles, cutoff):
    """Split puzzles by a solver's knowledge cutoff and report grouping accuracy and
    win rate (overall) for each side.

    A puzzle is "till cutoff" when its date <= `cutoff` (the solver could have seen
    it during training) and "after cutoff" otherwise. Dates are ISO strings, so the
    plain string comparison is a correct date comparison.

    For an embedding model like Numberbatch this is bookkeeping — it can't memorize
    puzzles, so both sides should read about the same. For an LLM, a gap where
    till-cutoff is much stronger is evidence of memorization rather than reasoning;
    an equal drop on both sides instead points to puzzles simply getting harder.

    Returns {"cutoff", "till_cutoff", "after_cutoff"} and prints a short summary.
    """
    till = [p for p in puzzles if p["date"] <= cutoff]
    after = [p for p in puzzles if p["date"] > cutoff]
    result = {
        "cutoff": cutoff,
        "till_cutoff": _grouping_and_win(solve, till),
        "after_cutoff": _grouping_and_win(solve, after),
    }
    print(f"cutoff date: {cutoff}")
    for label, key in (("till cutoff ", "till_cutoff"), ("after cutoff", "after_cutoff")):
        m = result[key]
        if m["puzzles"] == 0:
            print(f"  {label} ({m['puzzles']:>4} puzzles):  — (no puzzles)")
        else:
            print(f"  {label} ({m['puzzles']:>4} puzzles):  "
                  f"grouping {m['grouping_accuracy']:.3f}   win rate {m['win_rate']:.3f}")
    return result


def evaluate_recent(solve, puzzles, n=100):
    """Report grouping accuracy and win rate over the most recent `n` puzzles.

    `puzzles` is oldest-first (see load_puzzles), so the tail is the newest window.
    Read against the full-set numbers, a lower recent score is evidence that puzzles
    have gotten harder over time — a solver-agnostic trend that helps disentangle a
    knowledge-cutoff gap (see evaluate_by_cutoff) from rising difficulty.

    Returns {"window", "recent"} and prints a short summary.
    """
    recent = puzzles[-n:]
    result = {"window": len(recent), "recent": _grouping_and_win(solve, recent)}
    m = result["recent"]
    print(f"most recent {m['puzzles']} puzzles ({recent[0]['date']} .. {recent[-1]['date']}):"
          if recent else f"most recent {n} puzzles:")
    if m["puzzles"] == 0:
        print("  — (no puzzles)")
    else:
        print(f"  grouping {m['grouping_accuracy']:.3f}   win rate {m['win_rate']:.3f}")
    return result


def evaluate(name, solve, puzzles, show_miss_distance=True):
    """Run `solve(words) -> 4 groups` over every puzzle, print the metrics, and
    save a summary + per-puzzle breakdown to results/{name}-{timestamp}.json.

    Reports grouping accuracy (groups solved overall), per-color accuracy, and
    win rate (puzzles where all 4 groups were solved).

    show_miss_distance=False drops the miss-distance histogram (printed and saved). Pass
    it for a solver that substitutes a placeholder partition when it can't answer: the
    distance would describe the placeholder rather than the solver.
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
        missed = miss_distance(guess, puzzle["answer"], COLORS) if show_miss_distance else None
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
    miss_dist = (miss_distance_distribution(per_puzzle_missed, COLORS)
                 if show_miss_distance else None)

    print(f"puzzles:           {n}")
    print(f"grouping accuracy: {grouping_accuracy:.3f}  ({total_hits}/{n * 4} groups)")
    print("color accuracy:")
    for color in COLORS:
        print(f"  {color:<8} {color_hits[color] / n:.3f}")
    print(f"win rate:          {wins / n:.3f}")
    if show_miss_distance:
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
