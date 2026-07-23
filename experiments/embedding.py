# Word2vec baseline: embed the 16 words with GoogleNews-vectors-negative300,
# then brute-force the 4-group partition that maximizes pairwise cosine
# similarity. Run with the repo venv (anaconda's gensim is broken):
#   venv/bin/python experiments/embedding.py
# Model auto-downloads via kagglehub on first run (~3.6GB, cached after).

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from connections import RESULTS_DIR, evaluate, load_puzzles

# --- config, echoed into every result for reproducibility ---
CONFIG = {
    "embedding": "GoogleNews-vectors-negative300",
    "preprocess": "raw vectors; three-case fallback; phrase = underscore token else content-word average",
    "similarity": "cosine",
    "objective": "mean cosine similarity to group centroid",
    "search": "exact brute-force over all 2,627,625 partitions",
}

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "for"}


# --- vector lookup ---------------------------------------------------------

def _lookup(kv, token):
    """First hit wins: exact -> title case -> lowercase. None if all miss."""
    for form in (token, token.title(), token.lower()):
        if form in kv:
            return kv[form], form
    return None, None


def embed(kv, word):
    """Vector for one puzzle word plus the fallback tier that resolved it.

    Single token: three-case fallback. Phrase: underscore-joined token first,
    else the average of its content-word vectors (stopwords dropped).
    Returns (vector, tier) where tier is exact|title|lower|underscore|averaged|oov.
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


# --- similarity ------------------------------------------------------------

def similarity_matrix(vectors):
    """16x16 cosine similarity. Zero-norm (OOV) vectors give similarity 0."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return unit @ unit.T


# --- objective + search -----------------------------------------------------
#
# A group objective is (sim, group) -> float, scoring one group of 4 in
# isolation. Swap objectives by passing a different one into search() (e.g. a
# future min_pairwise_score, centroid_score) — see search()'s docstring for
# the constraint this places on new objectives.

def sum_pairwise_score(sim, group):
    """Sum of the 6 pairwise similarities within the group."""
    return sim[np.ix_(group, group)][np.triu_indices(len(group), k=1)].sum()


def min_pairwise_score(sim, group):
    """Weakest of the 6 pairwise similarities within the group."""
    return sim[np.ix_(group, group)][np.triu_indices(len(group), k=1)].min()


def centroid_score(vectors, group):
    """Mean cosine similarity of the group's members to their own centroid.

    Needs raw vectors, not just `sim` — pass search(vectors, centroid_score)
    instead of search(sim, ...) when using this objective.
    """
    members = vectors[np.array(group)]
    centroid = members.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return 0.0
    member_norms = np.linalg.norm(members, axis=1)
    sims = np.divide(
        members @ centroid, member_norms * norm,
        out=np.zeros(len(members)), where=member_norms != 0,
    )
    return sims.mean()


def _partitions(items):
    """Every way to split items into groups of 4, each emitted exactly once.

    Canonical order: the lowest-indexed unassigned item always leads its group.
    """
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for combo in itertools.combinations(rest, 3):
        remaining = [x for x in rest if x not in combo]
        for tail in _partitions(remaining):
            yield [(first,) + combo] + tail


# Partition structure is identical across puzzles, only scores change — build
# it once. GROUPS: the 1,820 possible 4-subsets. PARTITION_IDX: (2,627,625 x 4)
# indices into GROUPS, one row per partition, in canonical order (so ties
# resolve to the lexicographically smallest partition via first-argmax).
GROUPS = list(itertools.combinations(range(16), 4))
_GROUP_ID = {g: i for i, g in enumerate(GROUPS)}
PARTITION_IDX = np.array(
    [[_GROUP_ID[tuple(g)] for g in p] for p in _partitions(list(range(16)))],
    dtype=np.int32,
)


def search(data, group_score_fn=sum_pairwise_score):
    """Exact argmax partition, returned as a tuple of 4 sorted-index tuples.

    `data` is whatever group_score_fn needs (the sim matrix for
    sum/min_pairwise_score, raw vectors for centroid_score) — search() never
    looks inside it, just forwards it to group_score_fn(data, group).

    Requires a group-additive objective — a group's score must not depend on
    the other 3 groups in its partition. That lets us score all 1,820 possible
    groups once, then get all 2.6M partition scores via one vectorized
    index-and-sum, instead of rescoring every partition from scratch.

    Not usable for within-vs-between-style objectives, where a group's score
    depends on which other groups it's paired with."""
    group_score = np.array([group_score_fn(data, g) for g in GROUPS])
    best = np.argmax(group_score[PARTITION_IDX].sum(axis=1))
    partition = tuple(GROUPS[i] for i in PARTITION_IDX[best])
    return partition, float(group_score[PARTITION_IDX[best]].sum())


# --- solver ----------------------------------------------------------------

_KV = None


def _load_model():
    global _KV
    if _KV is None:
        import kagglehub
        from gensim.models import KeyedVectors

        path = Path(kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300"))
        bin_path = next(path.rglob("*.bin"))
        _KV = KeyedVectors.load_word2vec_format(str(bin_path), binary=True)
    return _KV


def solve_detailed(words):
    """Solve one puzzle and return the full per-puzzle record from the spec."""
    kv = _load_model()
    start = time.perf_counter()

    vectors, tiers = zip(*(embed(kv, w) for w in words))
    vectors = np.array(vectors, dtype=np.float32)
    oov = [w for w, t in zip(words, tiers) if t == "oov"]

    partition, best_score = search(vectors, centroid_score)
    groups = [[words[i] for i in g] for g in partition]

    return {
        "predicted_groups": groups,
        "best_score": best_score,
        "oov_words": oov,
        "has_oov": bool(oov),
        "lookup_tiers": dict(zip(words, tiers)),
        "runtime_sec": time.perf_counter() - start,
        **CONFIG,
    }


def solve(words):
    """Harness contract: 16 words -> 4 groups of 4."""
    return solve_detailed(words)["predicted_groups"]


if __name__ == "__main__":
    import json
    from datetime import datetime

    puzzles = load_puzzles()

    # Solve each puzzle once, keeping the full detail record. evaluate() re-calls
    # solve(words), so cache groups on the words tuple to avoid re-solving.
    records = [{"game_id": p["game_id"], "date": p["date"], **solve_detailed(p["words"])}
               for p in puzzles]
    cache = {tuple(p["words"]): r["predicted_groups"] for p, r in zip(puzzles, records)}
    cached_solve = lambda words: cache[tuple(words)]

    clean = [p for p, r in zip(puzzles, records) if not r["has_oov"]]

    print("=== all puzzles ===")
    evaluate("embedding", cached_solve, puzzles)
    print(f"\n=== has_oov=False subset ({len(clean)}/{len(puzzles)}) ===")
    evaluate("embedding_no_oov", cached_solve, clean)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = RESULTS_DIR / f"embedding-detail-{stamp}.json"
    detail_file.write_text(json.dumps({"config": CONFIG, "puzzles": records}, indent=2))
    print(f"\nSaved per-puzzle detail to {detail_file}")
