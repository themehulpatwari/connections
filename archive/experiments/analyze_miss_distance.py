# Temporary analysis script: for each color, across ALL puzzles, find the
# predicted group with the best word-overlap against that color's true group,
# and bucket puzzles by how many words were "missed" (4 - overlap).
# missed=0 means the group was solved exactly.
#
# Usage: python3 experiments/analyze_miss_distance.py [detail_json_path]

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from connections import COLORS, load_puzzles  # noqa: E402

DETAIL_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "results" / "embedding-detail-20260725-101352.json"
)
OUT_FILE = ROOT / "results" / "miss_distance_by_color.json"


def best_overlap_miss(true_group, guess):
    """Return (missed_count, best_predicted_group) for the predicted group
    that shares the most words with true_group."""
    true_set = set(true_group)
    best = max(guess, key=lambda g: len(true_set & set(g)))
    overlap = len(true_set & set(best))
    return 4 - overlap, best


def main():
    detail = json.loads(DETAIL_FILE.read_text())
    puzzles_by_id = {p["game_id"]: p for p in load_puzzles()}

    # color -> missed_count (0..4) -> count of puzzles
    buckets = {color: {n: 0 for n in range(5)} for color in COLORS}
    per_puzzle_detail = {color: [] for color in COLORS}

    for entry in detail["puzzles"]:
        game_id = entry["game_id"]
        puzzle = puzzles_by_id[game_id]
        answer = puzzle["answer"]
        guess = entry["predicted_groups"]

        for i, color in enumerate(COLORS):
            true_group = answer[i]
            missed, best = best_overlap_miss(true_group, guess)
            buckets[color][missed] += 1
            per_puzzle_detail[color].append({
                "game_id": game_id,
                "date": entry["date"],
                "missed": missed,
                "true_group": true_group,
                "closest_predicted_group": best,
            })

    n = len(detail["puzzles"])
    summary = {}
    for color in COLORS:
        dist = buckets[color]
        summary[color] = {
            "n": n,
            "missed_0_solved": {"count": dist[0], "pct": round(dist[0] / n * 100, 1)},
            "missed_1": {"count": dist[1], "pct": round(dist[1] / n * 100, 1)},
            "missed_2": {"count": dist[2], "pct": round(dist[2] / n * 100, 1)},
            "missed_3": {"count": dist[3], "pct": round(dist[3] / n * 100, 1)},
            "missed_4": {"count": dist[4], "pct": round(dist[4] / n * 100, 1)},
        }

    print(f"puzzles: {n}\n")
    header = f"{'color':<8} {'missed=0 (solved)':>18} {'missed=1':>10} {'missed=2':>10} {'missed=3':>10} {'missed=4':>10}"
    print(header)
    for color in COLORS:
        s = summary[color]
        print(
            f"{color:<8} "
            f"{s['missed_0_solved']['count']:>6} ({s['missed_0_solved']['pct']:>4.1f}%)   "
            f"{s['missed_1']['count']:>4} ({s['missed_1']['pct']:>4.1f}%) "
            f"{s['missed_2']['count']:>4} ({s['missed_2']['pct']:>4.1f}%) "
            f"{s['missed_3']['count']:>4} ({s['missed_3']['pct']:>4.1f}%) "
            f"{s['missed_4']['count']:>4} ({s['missed_4']['pct']:>4.1f}%)"
        )

    OUT_FILE.write_text(json.dumps(
        {"summary": summary, "per_puzzle": per_puzzle_detail}, indent=2
    ))
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
