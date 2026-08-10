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

The current solver is the ConceptNet Numberbatch baseline (raw vectors, cosine
similarity, sum-of-pairwise-similarity score, exact brute-force search):

```bash
venv/bin/python experiments/numberbatch.py
```

It prints grouping accuracy — the fraction of groups it got exactly right —
across the dataset, plus win rate and the oracle rank of the true partition.

Earlier exploratory work (the swappable embedding/sentence-embedding sweeps, the
GloVe/fastText/word2vec loaders, and the analysis scripts) lives in `archive/`
and still runs in place, e.g. `venv/bin/python archive/experiments/embedding.py`.

## Add your own

Copy any file in `experiments/`, write a `solve(words)` that returns four groups
of four words, and run it. Nothing else needs to change.
