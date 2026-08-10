# Temporary analysis script: find puzzles where the embedding solver got at
# least one group right, and dump (a) the predicted groups with solved/unsolved
# color labels and (b) the true answer groups, sorted by #groups solved desc.
#
# Usage: python3 experiments/analyze_partial_solves.py [detail_json_path]

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from connections import COLORS, load_puzzles, solved_colors  # noqa: E402

DETAIL_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "results" / "embedding-detail-20260725-101352.json"
)
OUT_PREDICTED = ROOT / "results" / "partial_solves_predicted.json"
OUT_ANSWERS = ROOT / "results" / "partial_solves_answers.json"


def main():
    detail = json.loads(DETAIL_FILE.read_text())
    puzzles_by_id = {p["game_id"]: p for p in load_puzzles()}

    predicted_out = []
    answers_out = []

    for entry in detail["puzzles"]:
        game_id = entry["game_id"]
        puzzle = puzzles_by_id[game_id]
        answer = puzzle["answer"]
        guess = entry["predicted_groups"]

        solved = solved_colors(guess, answer)
        if not solved:
            continue  # only keep puzzles with at least one correct group

        num_solved = len(solved)

        # Map each predicted group to its status: the color it solved, or "wrong".
        answer_sets = {frozenset(g): COLORS[i] for i, g in enumerate(answer)}
        predicted_groups_labeled = []
        for group in guess:
            color = answer_sets.get(frozenset(group))
            predicted_groups_labeled.append({
                "words": group,
                "status": color if color else "wrong",
            })

        predicted_out.append({
            "game_id": game_id,
            "date": entry["date"],
            "num_solved": num_solved,
            "solved_colors": sorted(solved, key=COLORS.index),
            "unsolved_colors": [c for c in COLORS if c not in solved],
            "predicted_groups": predicted_groups_labeled,
        })

        answers_out.append({
            "game_id": game_id,
            "date": entry["date"],
            "num_solved": num_solved,
            "solved_colors": sorted(solved, key=COLORS.index),
            "answer_groups": [
                {"color": COLORS[i], "words": g, "was_solved": COLORS[i] in solved}
                for i, g in enumerate(answer)
            ],
        })

    # Sort by num_solved descending (4 -> 3 -> 2 -> 1), stable on game_id.
    predicted_out.sort(key=lambda x: (-x["num_solved"], x["game_id"]))
    answers_out.sort(key=lambda x: (-x["num_solved"], x["game_id"]))

    OUT_PREDICTED.write_text(json.dumps(predicted_out, indent=2))
    OUT_ANSWERS.write_text(json.dumps(answers_out, indent=2))

    counts = {n: sum(1 for p in predicted_out if p["num_solved"] == n) for n in (4, 3, 2, 1)}
    print(f"Puzzles with >=1 group solved: {len(predicted_out)}")
    print(f"  4 solved (win): {counts[4]}")
    print(f"  3 solved:       {counts[3]}")
    print(f"  2 solved:       {counts[2]}")
    print(f"  1 solved:       {counts[1]}")
    print(f"\nPredicted groups (with solved/unsolved labels): {OUT_PREDICTED}")
    print(f"True answer groups (for same puzzles):           {OUT_ANSWERS}")


if __name__ == "__main__":
    main()
