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
from connections import (
    RESULTS_DIR, evaluate, evaluate_by_cutoff, evaluate_recent, load_puzzles,
)

DATA_DIR = Path(__file__).parent.parent / "data"
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "for"}

# Knowledge cutoff for the till/after-cutoff metric. ConceptNet Numberbatch 19.08 was
# built in August 2019, so that release date is this solver's "cutoff". Every current
# puzzle postdates it, so the whole set is after-cutoff. Bookkeeping only — an embedding
# model can't memorize puzzles — but it's the same control the LLM solvers reuse, where
# a till-vs-after gap would separate memorization from puzzles just getting harder.
CUTOFF_DATE = "2019-08-01"


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


def partition_overlap(a_ids, b_ids):
    """Word-level overlap between two partitions, each given as 4 group-ids into
    GROUPS: the max, over all 24 pairings of a's groups to b's groups, of the total
    words sharing a group. 16 = identical partition; 14 = one word-swap; lower =
    more scrambled relative to each other."""
    a = [set(GROUPS[i]) for i in a_ids]
    b = [set(GROUPS[i]) for i in b_ids]
    return max(
        sum(len(a[i] & b[perm[i]]) for i in range(4))
        for perm in itertools.permutations(range(4))
    )


# Rank band whose above-truth competitors we dump for diagnostics: truth is close
# (in the top 100) but not the argmax. Rank-1 puzzles have nothing above truth;
# beyond 100 truth isn't really "almost found", so we don't enumerate those.
ABOVE_TRUTH_BAND = (2, 100)


def search(sim, truth=None):
    """Exact argmax partition (4 sorted-index tuples), plus — when `truth` is a
    PARTITION_IDX row — how many partitions outscore the true one.

    Score all 1,820 groups once, then get all 2.6M partition scores via one
    vectorized index-and-sum. `truth` is measurement only: the returned partition
    is argmax(scores) and cannot depend on it.

    For puzzles in ABOVE_TRUTH_BAND, diagnostics also carries `above_truth`: every
    partition that outscores truth, best-first, with its score and word-level overlap
    with truth (partition given as 4 group-id tuples, mapped to words by the caller).
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
        rank = int((scores > truth_score).sum()) + 1
        diagnostics = {
            "oracle_rank": rank,
            "n_tied": int((scores == truth_score).sum()) - 1,
        }
        if ABOVE_TRUTH_BAND[0] <= rank <= ABOVE_TRUTH_BAND[1]:
            above = np.flatnonzero(scores > truth_score)
            above = above[np.argsort(-scores[above], kind="stable")]  # best-first
            truth_ids = PARTITION_IDX[truth]
            diagnostics["above_truth"] = [
                {
                    "rank": i + 1,
                    "score": float(scores[idx]),
                    "overlap": partition_overlap(PARTITION_IDX[idx], truth_ids),
                    "group_indices": [GROUPS[g] for g in PARTITION_IDX[idx]],
                }
                for i, idx in enumerate(above)
            ]
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

    # Map each above-truth competitor's group indices onto the puzzle's words.
    for entry in diagnostics.get("above_truth", []):
        entry["groups"] = [[words[i] for i in g] for g in entry.pop("group_indices")]

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

    print("\n=== knowledge-cutoff split ===")
    cutoff_summary = evaluate_by_cutoff(cached_solve, puzzles, CUTOFF_DATE)

    print("\n=== recent-window (has puzzles gotten harder?) ===")
    recent_summary = evaluate_recent(cached_solve, puzzles, n=100)

    # oracle_rank: where the true partition ranks among all 2,627,625 (all puzzles).
    ranks = np.array([r["oracle_rank"] for r in records])
    q1, med, q3 = np.percentile(ranks, [25, 50, 75])
    topk = {k: float((ranks <= k).mean()) for k in (1, 5, 10, 20, 50, 100)}
    oracle_summary = {
        "median": float(med), "q1": float(q1), "q3": float(q3),
        "rank_1": topk[1], "top_100": topk[100], "topk_curve": topk,
    }
    print("\noracle rank (truth's rank among 2,627,625 partitions):")
    print(f"  median {med:.0f}  (q1 {q1:.0f}, q3 {q3:.0f})")
    print("  top-k curve (fraction of puzzles with truth ranked <= k):")
    print("    " + "   ".join(f"k={k}:{topk[k]:.3f}" for k in (1, 5, 10, 20, 50, 100)))

    # above-truth diagnostics: for puzzles where truth is top-100 but not rank-1, look
    # at the partitions that outscore it. overlap 16 = identical, 14 = one swap; a band
    # dominated by high overlaps means truth is losing to near-misses, not to wholly
    # different groupings. We summarize the single best competitor per puzzle plus the
    # overlap distribution across all competitors (full dump lives in the detail JSON).
    band = [r for r in records if "above_truth" in r]
    if band:
        top_overlaps = np.array([r["above_truth"][0]["overlap"] for r in band])
        all_overlaps = np.array([e["overlap"] for r in band for e in r["above_truth"]])
        n_competitors = len(all_overlaps)
        overlap_hist = {int(v): int((all_overlaps == v).sum()) for v in np.unique(all_overlaps)}
        oracle_summary["above_truth"] = {
            "band": list(ABOVE_TRUTH_BAND),
            "n_puzzles": len(band),
            "n_competitors": n_competitors,
            "top_competitor_overlap_mean": float(top_overlaps.mean()),
            "overlap_hist": overlap_hist,
        }
        print(f"\nabove-truth (rank {ABOVE_TRUTH_BAND[0]}-{ABOVE_TRUTH_BAND[1]}): "
              f"{len(band)} puzzles, {n_competitors} competing partitions")
        print(f"  best-competitor overlap with truth (of 16): "
              f"mean {top_overlaps.mean():.1f}, min {top_overlaps.min()}, max {top_overlaps.max()}")
        print("  overlap histogram over all competitors (words in a correct group, of 16):")
        print("    " + "   ".join(f"{v}:{overlap_hist[v]}" for v in sorted(overlap_hist)))
    else:
        oracle_summary["above_truth"] = {"band": list(ABOVE_TRUTH_BAND), "n_puzzles": 0}
        print(f"\nabove-truth (rank {ABOVE_TRUTH_BAND[0]}-{ABOVE_TRUTH_BAND[1]}): no puzzles in band")

    print(f"\n=== has_oov=False subset ({len(clean)}/{len(puzzles)}) ===")
    evaluate("numberbatch_no_oov", cached_solve, clean)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = RESULTS_DIR / f"numberbatch-detail-{stamp}.json"
    detail_file.write_text(json.dumps(
        {"config": CONFIG, "oracle_rank": oracle_summary,
         "cutoff": cutoff_summary, "recent": recent_summary, "puzzles": records}, indent=2))
    print(f"\nSaved per-puzzle detail to {detail_file}")
