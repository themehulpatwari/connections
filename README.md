# connections

Benchmarking approaches to solving [NYT Connections](https://www.nytimes.com/games/connections) —
sort 16 words into 4 hidden groups of 4.

Each approach is a small standalone script in `experiments/`. They all share two
helpers from `connections.py`: `load_puzzles()` and `score()`. That's the whole
framework — a solver is just a `solve(words) -> 4 groups of 4` function.

## Setup

```bash
pip install -r requirements.txt
python data.py            # download + build data/Connections_Data.csv
```

## Run an experiment

```bash
python experiments/random_baseline.py   # shuffle baseline (the floor)
python experiments/embedding.py     # (todo)
python experiments/llm_zeroshot.py  # (todo)
python experiments/llm_cot.py       # (todo)
```

Each prints grouping accuracy — the fraction of groups it got exactly right —
across the dataset.

## Add your own

Copy any file in `experiments/`, write a `solve(words)` that returns four groups
of four words, and run it. Nothing else needs to change.
