# Abstract base interface that all solvers implement.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.puzzle import Puzzle

if TYPE_CHECKING:
    from src.game import GameState


class Solver(ABC):
    """Names four words at a time until the game ends.

    This is the only interface the game harness knows about. Solvers that adapt
    to feedback read it off the state; solvers that plan upfront should subclass
    StaticSolver instead.
    """

    name: str

    @abstractmethod
    def next_guess(self, state: GameState) -> list[str]:
        """Return the next four words to submit, or an empty list to resign."""

    def reset(self) -> None:
        """Discard per-puzzle state. Called by the harness before each game."""


class StaticSolver(Solver):
    """Base for solvers that plan the whole board once and ignore feedback."""

    def __init__(self) -> None:
        self._plan: list[list[str]] | None = None

    @abstractmethod
    def solve(self, puzzle: Puzzle) -> list[list[str]]:
        """Return 4 groups of 4 words. Order doesn't matter — scoring checks
        whether each group was found at all, not when it was guessed."""

    def reset(self) -> None:
        self._plan = None

    def next_guess(self, state: GameState) -> list[str]:
        if self._plan is None:
            self._plan = [list(group) for group in self.solve(state.puzzle)]
        return self._plan.pop(0) if self._plan else []
