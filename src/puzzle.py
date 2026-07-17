# Shared types describing a single Connections puzzle.

from dataclasses import dataclass
from datetime import date

# NYT colors, easiest to hardest.
LEVEL_COLORS = ("yellow", "green", "blue", "purple")


@dataclass(frozen=True)
class Group:
    """One of the four hidden categories in a puzzle."""

    name: str
    level: int
    words: tuple[str, ...]

    @property
    def color(self) -> str:
        return LEVEL_COLORS[self.level]


@dataclass(frozen=True)
class Puzzle:
    """A single day's puzzle: 16 words in board order and the 4 groups they belong to."""

    game_id: int
    date: date
    words: tuple[str, ...]
    groups: tuple[Group, ...]

    def group_for(self, word: str) -> Group:
        """Return the group the given word belongs to."""
        for group in self.groups:
            if word in group.words:
                return group
        raise KeyError(f"{word!r} is not in puzzle {self.game_id}")
