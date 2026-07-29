# Miss-distance metric: for each color's true group, how many words did the
# solver's closest-matching guessed group miss (0 = solved exactly).


def best_overlap_miss(true_group, guess):
    """Return (missed_count, best_guessed_group) for the guessed group that
    shares the most words with true_group. missed_count is 4 - overlap."""
    true_set = set(true_group)
    best = max(guess, key=lambda g: len(true_set & set(g)))
    overlap = len(true_set & set(best))
    return 4 - overlap, best


def miss_distance(guess, answer, colors):
    """Per-color miss distance for one puzzle.

    answer[i] is the colors[i] group. Returns {color: missed_count}.
    """
    return {
        colors[i]: best_overlap_miss(answer[i], guess)[0]
        for i in range(len(colors))
    }


def miss_distance_distribution(per_puzzle_missed, colors):
    """Aggregate a list of per-puzzle {color: missed_count} dicts into
    {color: {missed_count (0-4): count}}."""
    dist = {color: {n: 0 for n in range(5)} for color in colors}
    for entry in per_puzzle_missed:
        for color, missed in entry.items():
            dist[color][missed] += 1
    return dist
