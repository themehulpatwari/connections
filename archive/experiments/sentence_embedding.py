# Sentence-embedding solver (Qwen3-Embedding, Jina, mpnet, gemini). Unlike the
# static-embedding path this encodes entries with a model that sees context, so
# two things change: there is no OOV (subword tokenization always produces a
# vector, so compare against GloVe's no-OOV number, not its full-set number),
# and entry order can affect the result.
#
# Encoding is not implemented yet — encode_all() is the one missing piece.
# Everything downstream of it works and is tested.
#
#   venv/bin/python experiments/sentence_embedding.py --preset qwen_bare
#   venv/bin/python experiments/sentence_embedding.py --preset qwen_bare --preprocess all_but_top_k
#   venv/bin/python experiments/sentence_embedding.py --preset qwen_bare --oracle-rank

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from connections import RESULTS_DIR, evaluate, load_puzzles
from embedding import (GROUPS, PARTITION_IDX, centroid_score, min_pairwise_score,
                       sum_pairwise_score)

CSLS_K = 10


# --- config ------------------------------------------------------------------

@dataclass
class Config:
    """Every lever for one run, so an experiment is a one-field edit.

    Note on the GloVe numbers this is measured against: `all_but_top_k` and
    `mean_center` results there were computed with OOV zero-vectors riding
    through the transform, so full-set numbers for those configs aren't directly
    comparable to `raw`. It doesn't apply here — no OOV — but it explains the
    historical numbers.
    """

    embedding: str                  # HF model id (gemini-embedding-001 is an API id, not HF)
    input_format: str = "bare"      # bare | template | instructed | shared_context
    instruction: str | None = None  # input_format == "instructed" only
    template: str = "The word {word}."  # input_format == "template" only; {word} = lowercased entry
    n_permutations: int = 1         # input_format == "shared_context" only
    task: str | None = None         # jina-embeddings-v5 task adapter (e.g. "clustering"); other models ignore it
    normalize: bool = True
    preprocess: str = "raw"         # raw | l2_normalize | mean_center | all_but_top_k
    pca_k: int = 3
    similarity: str = "cosine"      # cosine | csls
    objective: str = "sum_pairwise" # sum_pairwise | min_pairwise | centroid
    search: str = "exact"
    shuffle_seed: int = 0
    batch_size: int = 16


QWEN = "Qwen/Qwen3-Embedding-0.6B"  # the tier's test model (not 8B; see the phase plan)
INSTRUCTION = (
    "Represent this word so that words belonging to the same hidden category "
    "in a NYT Connections puzzle are close together."
)

PRESETS = {
    "qwen_bare": Config(embedding=QWEN),
    "qwen_template": Config(embedding=QWEN, input_format="template"),
    "qwen_instructed": Config(embedding=QWEN, input_format="instructed", instruction=INSTRUCTION),
    "qwen_shared_context": Config(embedding=QWEN, input_format="shared_context", n_permutations=8),
    "qwen_8b_shared_context": Config(embedding="Qwen/Qwen3-Embedding-8B", input_format="shared_context", n_permutations=1),
    "jina_bare": Config(embedding="jinaai/jina-embeddings-v5-text-small", task="clustering"),
    "mpnet_bare": Config(embedding="sentence-transformers/all-mpnet-base-v2"),
    "mpnet_shared_context": Config(embedding="sentence-transformers/all-mpnet-base-v2", input_format="shared_context", n_permutations=8),
}


# --- input formats -----------------------------------------------------------
# The input_format lever, swappable by config (or --input-format) alone. Three
# formats map each entry to one text and let the model card pool it (see
# _encode_pooled); shared_context encodes all 16 in one pass (_encode_shared_context).

INPUT_FORMATS = ("bare", "template", "instructed", "shared_context")

# Qwen3-Embedding query format. Verify against the model card before trusting it;
# dry_run.py prints the resolved prefix, and it is logged per run in the output.
_INSTRUCT = "Instruct: {task}\nQuery:{text}"


def _pooled_texts(words, config):
    """The 16 resolved inputs for bare/template/instructed (entries lowercased)."""
    entries = [w.lower() for w in words]
    if config.input_format == "bare":
        return entries
    if config.input_format == "template":
        return [config.template.format(word=e) for e in entries]
    if config.input_format == "instructed":
        return [_INSTRUCT.format(task=config.instruction, text=e) for e in entries]
    raise ValueError(f"not a pooled format: {config.input_format}")


def _orderings(config, puzzle):
    """n_permutations entry orderings for shared_context. The first is the entries
    as loaded (already shuffled by load_shuffled_puzzles); the rest are seeded
    permutations, so the averaging is reproducible across models and runs."""
    base = np.arange(16)
    if config.n_permutations <= 1:
        return [base]
    rng = np.random.default_rng(config.shuffle_seed + puzzle["game_id"])
    return [base] + [rng.permutation(16) for _ in range(config.n_permutations - 1)]


def _shared_context_string(words, order):
    """All 16 entries, lowercased and comma-separated, in `order`."""
    return ", ".join(words[i].lower() for i in order)


def _resolved_inputs(puzzles, config):
    """Per-puzzle resolved model input(s), for logging: a list of 16 strings for the
    pooled formats, or the first-ordering comma-joined string for shared_context."""
    if config.input_format == "shared_context":
        return [_shared_context_string(p["words"], _orderings(config, p)[0]) for p in puzzles]
    return [_pooled_texts(p["words"], config) for p in puzzles]


# --- preprocess --------------------------------------------------------------
# (vectors, pool) -> vectors. Independent alternatives, not composed. `pool` is
# every entry vector in the run stacked (n_puzzles*16, d) — it stands in for the
# vocabulary matrix the GloVe path centered and fit PCA against, since a
# sentence model has no vocabulary to draw a reference distribution from.

def _unit(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


def raw(vectors, pool):
    return vectors


def l2_normalize(vectors, pool):
    return _unit(vectors)


def mean_center(vectors, pool):
    return vectors - pool.mean(axis=0)


_PCA_CACHE = {}


def _top_components(pool, k):
    """Top-k principal directions of the pool. Computed from the d x d Gram
    matrix rather than an SVD of the pool itself: d is up to 4096 here and only
    the top 3 directions are wanted, so the full SVD's U is pure waste."""
    key = (id(pool), k)
    if key not in _PCA_CACHE:
        centered = pool - pool.mean(axis=0)
        _, eigenvectors = np.linalg.eigh(centered.T @ centered)  # ascending
        _PCA_CACHE[key] = eigenvectors[:, ::-1][:, :k].T
    return _PCA_CACHE[key]


def all_but_top_k(vectors, pool, k=3):
    """Mean-center against the pool, then remove the projection onto the pool's
    top-k principal components (Mu & Viswanath's "all-but-the-top")."""
    centered = vectors - pool.mean(axis=0)
    components = _top_components(pool, k)
    return centered - (centered @ components.T) @ components


PREPROCESSORS = {
    "raw": raw,
    "l2_normalize": l2_normalize,
    "mean_center": mean_center,
    "all_but_top_k": all_but_top_k,
}


# --- similarity --------------------------------------------------------------
# (vectors, pool) -> 16x16 matrix. cosine ignores pool; csls draws its neighbor
# set from it. Independent alternatives, not composed.

def cosine_similarity_matrix(vectors, pool=None):
    unit = _unit(vectors)
    return unit @ unit.T


def csls_similarity_matrix(vectors, pool):
    """csls(a,b) = 2*cos(a,b) - r(a) - r(b), with r(w) the mean cosine
    similarity between w and its CSLS_K nearest neighbors in the pool.

    No r(w) disk cache, unlike the GloVe path: neighbors come from the run's own
    pool (~18k vectors, one matmul per puzzle) rather than a 500k-vector
    frequency band, so there is nothing expensive to amortize.
    """
    sims = _unit(vectors) @ _unit(pool).T
    # Each entry vector is itself in the pool at similarity 1.0 — drop that
    # self-match before taking the top CSLS_K.
    neighbors = np.sort(sims, axis=1)[:, -(CSLS_K + 1):-1]
    r = neighbors.mean(axis=1)
    return 2 * cosine_similarity_matrix(vectors) - r[:, None] - r[None, :]


SIMILARITIES = {"cosine": cosine_similarity_matrix, "csls": csls_similarity_matrix}

# (data, group) -> float, reused as-is from the GloVe path. sum/min_pairwise
# take a similarity matrix; centroid takes raw vectors (see solve_detailed).
OBJECTIVES = {
    "sum_pairwise": sum_pairwise_score,
    "min_pairwise": min_pairwise_score,
    "centroid": centroid_score,
}


# --- data --------------------------------------------------------------------

def load_shuffled_puzzles(shuffle_seed):
    """Puzzles with entries permuted, seeded per puzzle so the permutation is
    reproducible and identical across every model and config.

    Required, not hygiene: the stored order is the NYT board layout, which
    separates group members far more often than chance — side-by-side entries
    share a group 3.7% of the time against a 20% chance rate (z = -47.6, see
    scripts/check_word_order.py). Anything that reads the entries in order
    would be reading that.
    """
    puzzles = load_puzzles()
    for puzzle in puzzles:
        order = np.random.default_rng(shuffle_seed + puzzle["game_id"]).permutation(16)
        puzzle["words"] = [puzzle["words"][i] for i in order]
    return puzzles


# --- diagnostics -------------------------------------------------------------

_GROUP_ROW = {group: i for i, group in enumerate(GROUPS)}


def similarity_spread(sim):
    """Mean and std of the 120 distinct off-diagonal similarities. A narrow band
    means the objective is discriminating on very little dynamic range, which is
    what anisotropy looks like from here."""
    off_diagonal = sim[np.triu_indices(len(sim), k=1)]
    return float(off_diagonal.mean()), float(off_diagonal.std())


def truth_partition_row(words, answer):
    """The row of PARTITION_IDX holding the ground-truth partition."""
    position = {word: i for i, word in enumerate(words)}
    ids = sorted(_GROUP_ROW[tuple(sorted(position[w] for w in group))] for group in answer)
    return int(np.flatnonzero((PARTITION_IDX == ids).all(axis=1))[0])


# --- search ------------------------------------------------------------------

def search(data, group_score_fn, truth=None):
    """Exact argmax partition, plus — when `truth` is a PARTITION_IDX row — how
    many partitions outscore the true one.

    `truth` is measurement only. The returned partition comes from `scores`
    alone and cannot depend on it. Free to compute: every partition's score is
    already in `scores`, so this is a comparison, not a second search.
    """
    group_score = np.array([group_score_fn(data, group) for group in GROUPS])
    scores = group_score[PARTITION_IDX].sum(axis=1)
    best = int(np.argmax(scores))
    partition = tuple(GROUPS[i] for i in PARTITION_IDX[best])

    diagnostics = {}
    if truth is not None:
        # Read the true partition's score out of `scores` rather than re-summing
        # its four groups, so the tie test compares bit-identical floats.
        truth_score = scores[truth]
        diagnostics = {
            "oracle_rank": int((scores > truth_score).sum()) + 1,
            "n_tied": int((scores == truth_score).sum()) - 1,
        }
    return partition, float(scores[best]), diagnostics


# --- solver ------------------------------------------------------------------

# --- encoding ----------------------------------------------------------------
# encode -> normalize happens here; everything after (all_but_top_k -> cosine ->
# sum_pairwise -> exact search) is unchanged from the GloVe path. The model id is a
# plain config field, loaded but never branched on, so a Phase 2 model swap is a
# one-field edit. No OOV concept — subword tokenization always yields a vector.

CACHE_DIR = Path(__file__).parent.parent / "data" / "sentence_cache"

_ST_CACHE = {}
_HF_CACHE = {}


def _sentence_transformer(model_id):
    from sentence_transformers import SentenceTransformer
    if model_id not in _ST_CACHE:
        # trust_remote_code: Qwen3-Embedding / jina ship custom modeling code.
        _ST_CACHE[model_id] = SentenceTransformer(model_id, trust_remote_code=True)
    return _ST_CACHE[model_id]


def _auto_model(model_id):
    from transformers import AutoModel, AutoTokenizer
    if model_id not in _HF_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # .float(): Qwen3-Embedding checkpoints load as bfloat16, which numpy can't
        # consume; fp32 also keeps this path's precision matching the pooled one.
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).float()
        model.eval()
        _HF_CACHE[model_id] = (tok, model)
    return _HF_CACHE[model_id]


def _cache_path(config):
    safe = config.embedding.replace("/", "_")
    return CACHE_DIR / f"{safe}__{config.input_format}__seed{config.shuffle_seed}.npy"


def _encode_pooled(puzzles, config):
    """bare/template/instructed: one vector per entry, pooled by the model card
    (Qwen3-Embedding uses last-token pooling, not mean). Returns (n, 16, d)."""
    model = _sentence_transformer(config.embedding)
    texts = [t for p in puzzles for t in _pooled_texts(p["words"], config)]
    # task: jina-embeddings-v5 requires a task adapter; pass it only when set, so
    # models without one (Qwen, mpnet) aren't handed a kwarg their forward rejects.
    extra = {"task": config.task} if config.task else {}
    vecs = model.encode(
        texts, batch_size=config.batch_size,
        normalize_embeddings=config.normalize, convert_to_numpy=True, **extra,
    ).astype(np.float32)
    return vecs.reshape(len(puzzles), 16, -1)


def _entry_spans(text, entries):
    """Character (start, end) of each entry within the comma-joined text. Sequential
    (cursor advances past each match) so an entry that is a substring of another
    still lands on its own occurrence."""
    spans, cursor = [], 0
    for e in entries:
        start = text.index(e, cursor)
        spans.append((start, start + len(e)))
        cursor = start + len(e)
    return spans


def _pool_entries(text, entries, offsets, hidden):
    """Mean-pool each entry's final-layer token vectors. A token belongs to an entry
    if its character span overlaps the entry's; separators and special tokens (span
    width 0) fall outside every entry. Multi-word entries pool across their words,
    since the whole "ice cream" character span is a single entry span. Returns (16, d)."""
    out = np.zeros((len(entries), hidden.shape[1]), dtype=np.float32)
    tok_start, tok_end = offsets[:, 0], offsets[:, 1]
    for i, (s, e) in enumerate(_entry_spans(text, entries)):
        sel = (tok_end > tok_start) & (tok_start < e) & (tok_end > s)
        out[i] = hidden[sel].mean(axis=0)
    return out


def _encode_shared_context(puzzles, config):
    """All 16 entries in one forward pass, comma-separated in shuffled order.
    Mean-pool each entry's subword tokens from the final hidden layer, L2-normalize,
    average over n_permutations orderings, re-normalize. AutoModel/tokenizer directly
    (offsets map tokens back to entries), not SentenceTransformer. Returns (n, 16, d)."""
    import torch
    tok, model = _auto_model(config.embedding)
    d = model.config.hidden_size
    out = np.zeros((len(puzzles), 16, d), dtype=np.float32)
    for pi, p in enumerate(puzzles):
        orderings = _orderings(config, p)
        acc = np.zeros((16, d), dtype=np.float32)
        for order in orderings:
            entries = [p["words"][i].lower() for i in order]
            text = ", ".join(entries)
            enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].numpy()
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state[0].float().numpy()
            entry_vecs = _unit(_pool_entries(text, entries, offsets, hidden))
            acc[order] += entry_vecs   # order[j] is entry_vecs[j]'s canonical index
        out[pi] = _unit(acc / len(orderings))
    return out


def encode_all(puzzles, config):
    """Every puzzle's 16 entries -> ((n, 16, d) vectors, per-puzzle resolved inputs).

    Vectors are cached on disk keyed by (embedding, input_format, shuffle_seed).
    NOTE: the key excludes the template/instruction text, so if you change those,
    delete the stale cache file by hand. Resolved inputs are rebuilt every call
    (pure strings, no model).
    """
    resolved = _resolved_inputs(puzzles, config)
    path = _cache_path(config)
    if path.exists():
        return np.load(path), resolved
    encoder = _encode_shared_context if config.input_format == "shared_context" else _encode_pooled
    vectors = encoder(puzzles, config)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, resolved


def solve_detailed(puzzle, vectors, pool, config, resolved_input, oracle_rank=False):
    start = time.perf_counter()
    words = puzzle["words"]

    vectors = PREPROCESSORS[config.preprocess](vectors, pool)
    sim = SIMILARITIES[config.similarity](vectors, pool)
    # centroid scores vectors directly; the pairwise objectives score the matrix.
    data = vectors if config.objective == "centroid" else sim
    truth = truth_partition_row(words, puzzle["answer"]) if oracle_rank else None
    partition, best_score, diagnostics = search(data, OBJECTIVES[config.objective], truth)
    mean, std = similarity_spread(sim)

    return {
        "predicted_groups": [[words[i] for i in group] for group in partition],
        "entry_order": words,
        "resolved_input": resolved_input,
        "embedding_dim": int(vectors.shape[-1]),
        "oov_words": [],       # no OOV: subword tokenization always yields a vector
        "has_oov": False,
        "best_score": best_score,
        "similarity_mean": mean,
        "similarity_std": std,
        **diagnostics,
        "runtime_sec": time.perf_counter() - start,
        **asdict(config),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen_bare")
    parser.add_argument("--input-format", choices=sorted(INPUT_FORMATS))
    parser.add_argument("--preprocess", choices=sorted(PREPROCESSORS))
    parser.add_argument("--similarity", choices=sorted(SIMILARITIES))
    parser.add_argument("--objective", choices=sorted(OBJECTIVES))
    parser.add_argument("--oracle-rank", action="store_true",
                        help="also report the true partition's rank (diagnostic only)")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in {"input_format", "preprocess", "similarity", "objective"}}
    config = replace(PRESETS[args.preset], **overrides)
    puzzles = load_shuffled_puzzles(config.shuffle_seed)

    # encode_all normalizes (normalize_embeddings / manual L2); no re-normalize here.
    vectors, resolved = encode_all(puzzles, config)
    pool = vectors.reshape(-1, vectors.shape[-1])

    # evaluate() re-calls solve(words), so cache groups on the words tuple.
    records = [{"game_id": p["game_id"], "date": p["date"],
                **solve_detailed(p, v, pool, config, res, args.oracle_rank)}
               for p, v, res in zip(puzzles, vectors, resolved)]
    cache = {tuple(p["words"]): rec["predicted_groups"] for p, rec in zip(puzzles, records)}

    name = f"sentence_embedding-{args.preset}"
    summary = evaluate(name, lambda words: cache[tuple(words)], puzzles)

    n = len(puzzles)
    summary["games_won"] = int(round(summary["win_rate"] * n))
    summary["total_connections"] = int(round(summary["grouping_accuracy"] * n * 4))
    summary["similarity_mean"] = float(np.mean([r["similarity_mean"] for r in records]))
    summary["similarity_std"] = float(np.mean([r["similarity_std"] for r in records]))
    print(f"similarity spread: mean {summary['similarity_mean']:.3f}  "
          f"std {summary['similarity_std']:.3f}")

    if args.oracle_rank:
        ranks = np.array([r["oracle_rank"] for r in records])
        q1, med, q3 = np.percentile(ranks, [25, 50, 75])
        summary["oracle_rank_median"] = float(med)
        summary["oracle_rank_q1"] = float(q1)
        summary["oracle_rank_q3"] = float(q3)
        summary["oracle_rank_top_100"] = float((ranks <= 100).mean())
        print(f"oracle rank: median {summary['oracle_rank_median']:.0f} "
              f"(q1 {q1:.0f}, q3 {q3:.0f}) of {len(PARTITION_IDX):,}   "
              f"top-100 {summary['oracle_rank_top_100']:.3f}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = RESULTS_DIR / f"{name}-detail-{stamp}.json"
    detail_file.write_text(json.dumps(
        {"config": asdict(config), "summary": summary, "puzzles": records}, indent=2))
    print(f"\nSaved per-puzzle detail to {detail_file}")
