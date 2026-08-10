# NYT Connections solver, settled baseline config — hardcoded, no swappable knobs:
#   ConceptNet Numberbatch embeddings -> raw vectors -> cosine similarity ->
#   sum-of-pairwise-similarity group score -> exact brute-force search over all
#   2,627,625 partitions.
# Run with the repo venv (anaconda's gensim is broken):
#   venv/bin/python experiments/numberbatch.py
# Numberbatch auto-downloads (~280MB) on first run, cached in data/ after.

import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from connections import RESULTS_DIR, evaluate, load_puzzles

DATA_DIR = Path(__file__).parent.parent / "data"
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "for"}


# --- model -------------------------------------------------------------------

def load_numberbatch():
    """ConceptNet Numberbatch, English 19.08 — plain lowercase terms (underscores
    for multiword) in word2vec text format, downloaded from ConceptNet's S3 and
    cached in data/. gensim reads the .gz directly (no separate unzip)."""
    import urllib.request
    from gensim.models import KeyedVectors

    url = "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
    gz_path = DATA_DIR / "numberbatch-en-19.08.txt.gz"
    if not gz_path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Numberbatch (~280MB) to {gz_path} ...")
        urllib.request.urlretrieve(url, gz_path)
    return KeyedVectors.load_word2vec_format(str(gz_path))


# --- embedding ---------------------------------------------------------------

def _lookup(kv, token):
    """First hit wins: exact -> title case -> lowercase. None if all miss."""
    for form in (token, token.title(), token.lower()):
        if form in kv:
            return kv[form], form
    return None, None


def embed(kv, word):
    """Vector for one puzzle word plus the fallback tier that resolved it.

    Single token: three-case fallback. Phrase: underscore-joined token first, else
    the average of its content-word vectors (stopwords dropped). Returns
    (vector, tier) where tier is exact|title|lower|underscore|averaged|oov.
    """
    parts = word.replace("-", " ").split()
    if len(parts) == 1:
        vec, form = _lookup(kv, word)
        if vec is not None:
            tier = {word: "exact", word.title(): "title", word.lower(): "lower"}[form]
            return vec, tier
        return np.zeros(kv.vector_size, dtype=np.float32), "oov"

    vec, _ = _lookup(kv, "_".join(parts))
    if vec is not None:
        return vec, "underscore"

    content = [_lookup(kv, p)[0] for p in parts if p.lower() not in STOPWORDS]
    content = [v for v in content if v is not None]
    if content:
        return np.mean(content, axis=0), "averaged"
    return np.zeros(kv.vector_size, dtype=np.float32), "oov"


# --- scoring -----------------------------------------------------------------

def cosine_similarity_matrix(vectors):
    """16x16 cosine similarity. Zero-norm (OOV) vectors give similarity 0."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return unit @ unit.T


def sum_pairwise_score(sim, group):
    """Sum of the six pairwise similarities within a group of four."""
    return sim[np.ix_(group, group)][np.triu_indices(len(group), k=1)].sum()


# --- search ------------------------------------------------------------------

def _partitions(items):
    """Every way to split items into groups of 4, each emitted exactly once.
    Canonical: the lowest-indexed unassigned item always leads its group."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for combo in itertools.combinations(rest, 3):
        remaining = [x for x in rest if x not in combo]
        for tail in _partitions(remaining):
            yield [(first,) + combo] + tail


# Partition structure is the same for every puzzle, so build it once. GROUPS: the
# 1,820 possible 4-subsets. PARTITION_IDX: (2,627,625 x 4) indices into GROUPS, one
# row per partition, in canonical order — so the first argmax is already the
# lexicographically smallest partition (deterministic tie-break).
GROUPS = list(itertools.combinations(range(16), 4))
_GROUP_ID = {g: i for i, g in enumerate(GROUPS)}
PARTITION_IDX = np.array(
    [[_GROUP_ID[tuple(g)] for g in p] for p in _partitions(list(range(16)))],
    dtype=np.int32,
)


def truth_partition_row(words, answer):
    """Row of PARTITION_IDX holding the ground-truth partition, for oracle_rank."""
    position = {word: i for i, word in enumerate(words)}
    ids = sorted(_GROUP_ID[tuple(sorted(position[w] for w in group))] for group in answer)
    return int(np.flatnonzero((PARTITION_IDX == ids).all(axis=1))[0])


def search(sim, truth=None):
    """Exact argmax partition (4 sorted-index tuples), plus — when `truth` is a
    PARTITION_IDX row — how many partitions outscore the true one.

    Score all 1,820 groups once, then get all 2.6M partition scores via one
    vectorized index-and-sum. `truth` is measurement only: the returned partition
    is argmax(scores) and cannot depend on it.
    """
    group_score = np.array([sum_pairwise_score(sim, g) for g in GROUPS])
    scores = group_score[PARTITION_IDX].sum(axis=1)
    best = int(np.argmax(scores))
    partition = tuple(GROUPS[i] for i in PARTITION_IDX[best])

    diagnostics = {}
    if truth is not None:
        # Read truth's score out of `scores` (not a re-sum) so the tie test
        # compares bit-identical floats.
        truth_score = scores[truth]
        diagnostics = {
            "oracle_rank": int((scores > truth_score).sum()) + 1,
            "n_tied": int((scores == truth_score).sum()) - 1,
        }
    return partition, float(scores[best]), diagnostics


# --- solver ------------------------------------------------------------------

CONFIG = {
    "embedding": "numberbatch",
    "lookup": "three-case fallback; phrase = underscore token else content-word average",
    "preprocess": "raw",
    "similarity": "cosine_similarity_matrix",
    "objective": "sum_pairwise_score",
    "search": "exact brute-force over all 2,627,625 partitions",
}

_KV = None


def _load_model():
    global _KV
    if _KV is None:
        _KV = load_numberbatch()
    return _KV


def solve_detailed(words, answer=None):
    """Solve one puzzle and return the full per-puzzle record.

    When `answer` (the ground-truth grouping) is passed, also report oracle_rank /
    n_tied — where the true partition ranks among all 2,627,625. Diagnostic only;
    the returned partition is the argmax and never depends on it.
    """
    kv = _load_model()
    start = time.perf_counter()

    vectors, tiers = zip(*(embed(kv, w) for w in words))
    vectors = np.array(vectors, dtype=np.float32)
    oov = [w for w, t in zip(words, tiers) if t == "oov"]

    sim = cosine_similarity_matrix(vectors)
    truth = truth_partition_row(words, answer) if answer is not None else None
    partition, best_score, diagnostics = search(sim, truth)
    groups = [[words[i] for i in g] for g in partition]

    return {
        "predicted_groups": groups,
        "best_score": best_score,
        "oov_words": oov,
        "has_oov": bool(oov),
        "lookup_tiers": dict(zip(words, tiers)),
        **diagnostics,
        "runtime_sec": time.perf_counter() - start,
        **CONFIG,
    }


def solve(words):
    """Harness contract: 16 words -> 4 groups of 4."""
    return solve_detailed(words)["predicted_groups"]


if __name__ == "__main__":
    puzzles = load_puzzles()

    # Solve each puzzle once, keeping the full detail record. evaluate() re-calls
    # solve(words), so cache groups on the words tuple to avoid re-solving. Passing
    # the answer turns on oracle_rank (diagnostic only — see solve_detailed).
    records = [{"game_id": p["game_id"], "date": p["date"],
                **solve_detailed(p["words"], p["answer"])}
               for p in puzzles]
    cache = {tuple(p["words"]): r["predicted_groups"] for p, r in zip(puzzles, records)}
    cached_solve = lambda words: cache[tuple(words)]

    clean = [p for p, r in zip(puzzles, records) if not r["has_oov"]]

    print("=== all puzzles ===")
    evaluate("numberbatch", cached_solve, puzzles)

    # oracle_rank: where the true partition ranks among all 2,627,625 (all puzzles).
    ranks = np.array([r["oracle_rank"] for r in records])
    q1, med, q3 = np.percentile(ranks, [25, 50, 75])
    oracle_summary = {
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "rank_1": float((ranks == 1).mean()), "top_100": float((ranks <= 100).mean()),
    }
    print("oracle rank (truth's rank among 2,627,625 partitions):")
    print(f"  median {med:.0f}  (q1 {q1:.0f}, q3 {q3:.0f})")
    print(f"  rank-1 {oracle_summary['rank_1']:.3f}   top-100 {oracle_summary['top_100']:.3f}")

    print(f"\n=== has_oov=False subset ({len(clean)}/{len(puzzles)}) ===")
    evaluate("numberbatch_no_oov", cached_solve, clean)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = RESULTS_DIR / f"numberbatch-detail-{stamp}.json"
    detail_file.write_text(json.dumps(
        {"config": CONFIG, "oracle_rank": oracle_summary, "puzzles": records}, indent=2))
    print(f"\nSaved per-puzzle detail to {detail_file}")
