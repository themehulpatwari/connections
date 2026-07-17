# Core Connections game state and rules engine.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from src.puzzle import Group, Puzzle

if TYPE_CHECKING:
    from src.solvers.base import Solver

MAX_MISTAKES = 4
GUESS_SIZE = 4


@dataclass(frozen=True)
class Guess:
    """One submission and what the board said back."""

    words: tuple[str, ...]
    correct: bool
    one_away: bool
    group: Group | None


class GameState:
    """A game in progress: what's been solved, what's left, and the full transcript."""

    def __init__(self, puzzle: Puzzle, max_mistakes: int = MAX_MISTAKES) -> None:
        self.puzzle = puzzle
        self.max_mistakes = max_mistakes
        self.solved: list[Group] = []
        self.guesses: list[Guess] = []
        self.remaining_words: list[str] = list(puzzle.words)

    @property
    def mistakes(self) -> int:
        return sum(1 for guess in self.guesses if not guess.correct)

    @property
    def is_won(self) -> bool:
        return len(self.solved) == len(self.puzzle.groups)

    @property
    def is_over(self) -> bool:
        return self.is_won or self.mistakes >= self.max_mistakes

    def submit(self, words: Sequence[str]) -> Guess:
        """Submit four words, updating the board and returning the feedback."""
        words = tuple(words)
        if len(set(words)) != GUESS_SIZE:
            raise ValueError(f"A guess must be {GUESS_SIZE} distinct words, got {words!r}")
        off_board = set(words) - set(self.remaining_words)
        if off_board:
            raise ValueError(f"Words are not on the board: {sorted(off_board)}")

        unsolved = [group for group in self.puzzle.groups if group not in self.solved]
        overlaps = {group: len(set(words) & set(group.words)) for group in unsolved}
        matched = next((group for group, n in overlaps.items() if n == GUESS_SIZE), None)

        guess = Guess(
            words=words,
            correct=matched is not None,
            one_away=max(overlaps.values()) == GUESS_SIZE - 1,
            group=matched,
        )
        self.guesses.append(guess)

        if matched is not None:
            self.solved.append(matched)
            for word in words:
                self.remaining_words.remove(word)

        return guess


def play(puzzle: Puzzle, solver: Solver, max_mistakes: int = MAX_MISTAKES) -> GameState:
    """Play one puzzle to completion and return the finished state."""
    state = GameState(puzzle, max_mistakes=max_mistakes)
    solver.reset()

    while not state.is_over:
        guess = solver.next_guess(state)
        if not guess:
            break
        state.submit(guess)

    return state
