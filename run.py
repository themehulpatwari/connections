# Entry point for running solvers against puzzles and reporting results.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.data import load_puzzles
from src.game import play
from src.scoring import aggregate, score_game
from src.solvers.random_baseline import RandomSolver

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SOLVERS = {
    "random": RandomSolver,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a solver against NYT Connections puzzles.")
    parser.add_argument("--solver", choices=sorted(SOLVERS), default="random")
    parser.add_argument("--limit", type=int, help="Only run the most recent N puzzles.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for solvers that use randomness.")
    parser.add_argument("--save", action="store_true", help="Write per-puzzle results to results/.")
    args = parser.parse_args()

    puzzles = load_puzzles()
    if args.limit:
        puzzles = puzzles[-args.limit :]

    solver = SOLVERS[args.solver](seed=args.seed)
    states = [play(puzzle, solver) for puzzle in puzzles]
    summary = aggregate(states)

    print(f"solver:            {args.solver}")
    print(f"puzzles:           {summary['puzzles']}")
    print(f"grouping accuracy: {summary['grouping_accuracy']:.3f}")
    print("color accuracy:")
    for color, accuracy in summary["color_accuracy"].items():
        print(f"  {color:<8} {accuracy:.3f}")
    print(f"win rate:          {summary['win_rate']:.3f}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = RESULTS_DIR / f"{args.solver}-{stamp}.json"
        output_file.write_text(
            json.dumps(
                {
                    "solver": args.solver,
                    "seed": args.seed,
                    "summary": summary,
                    "games": [score_game(state) for state in states],
                },
                indent=2,
            )
        )
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
