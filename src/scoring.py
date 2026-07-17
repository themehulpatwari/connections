# Scoring and metrics for evaluating solver performance.

from __future__ import annotations

from statistics import mean

from src.game import GameState
from src.puzzle import LEVEL_COLORS


def score_game(state: GameState) -> dict:
    """Grade one finished game.

    Every metric is derived from the guess transcript, so it means the same
    thing whether the solver planned upfront or adapted to feedback.
    """
    solved_levels = {guess.group.level for guess in state.guesses if guess.correct}

    grouping_accuracy = len(solved_levels) / len(state.puzzle.groups)
    # Whether each color's group was found at all, regardless of guess order.
    color_accuracy = {color: level in solved_levels for level, color in enumerate(LEVEL_COLORS)}

    return {
        "game_id": state.puzzle.game_id,
        "date": state.puzzle.date.isoformat(),
        "grouping_accuracy": grouping_accuracy,
        "color_accuracy": color_accuracy,
        "won": state.is_won,
        "mistakes": state.mistakes,
    }


def aggregate(states: list[GameState]) -> dict:
    """Roll individual games up into benchmark-level metrics."""
    if not states:
        raise ValueError("Cannot aggregate an empty list of games")

    scores = [score_game(state) for state in states]

    return {
        "puzzles": len(scores),
        "grouping_accuracy": mean(score["grouping_accuracy"] for score in scores),
        "color_accuracy": {
            color: mean(score["color_accuracy"][color] for score in scores)
            for color in LEVEL_COLORS
        },
        "win_rate": mean(score["won"] for score in scores),
        "mean_mistakes": mean(score["mistakes"] for score in scores),
    }
