# Solver that partitions the board at random, as a floor for the other solvers.

from __future__ import annotations

import random

from src.puzzle import Puzzle
from src.solvers.base import StaticSolver


class RandomSolver(StaticSolver):
    """Shuffles the 16 words into 4 arbitrary groups."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def solve(self, puzzle: Puzzle) -> list[list[str]]:
        words = list(puzzle.words)
        self._rng.shuffle(words)
        return [words[i : i + 4] for i in range(0, 16, 4)]
