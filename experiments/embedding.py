# Word embedding baseline (word2vec / fastText / GloVe, see MODEL below). Run
# with the repo venv (anaconda's gensim is broken):
#   venv/bin/python experiments/embedding.py
# Model auto-downloads on first run (0.4-3.6GB depending on MODEL), cached after.

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from connections import RESULTS_DIR, evaluate, load_puzzles

STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "for"}

DATA_DIR = Path(__file__).parent.parent / "data"

# Both all_but_top_k's PCA and CSLS's neighbor search draw from this same pool:
# the top FREQ_BAND most-frequent words (GoogleNews vectors are frequency-
# sorted; this skips the long tail of rare/junk tokens past it). PCA samples
# randomly within the pool rather than using it whole, since the top few
# principal directions converge fast — verified stable, top-3 components
# agree at |cos| > 0.99 across two random-sample seeds.
FREQ_BAND = 500_000
PCA_SAMPLE_SIZE = 100_000
PCA_SEED = 0
CSLS_K = 10


# --- vector lookup -----------------------------------------------------------

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


# --- preprocess ----------------------------------------------------------------
# (vectors, model) -> vectors. Independent alternatives, not composed.

def raw(vectors, model=None):
    return vectors


def l2_normalize(vectors, model=None):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)


_VOCAB_MEAN_CACHE = {}


def _vocab_mean(model):
    key = id(model)
    if key not in _VOCAB_MEAN_CACHE:
        _VOCAB_MEAN_CACHE[key] = model.vectors.mean(axis=0)
    return _VOCAB_MEAN_CACHE[key]


def mean_center(vectors, model):
    return vectors - _vocab_mean(model)


_PCA_CACHE = {}


def _fit_pca_components(model, k, sample_size=None, seed=None):
    sample_size = PCA_SAMPLE_SIZE if sample_size is None else sample_size
    seed = PCA_SEED if seed is None else seed
    band = model.vectors[:FREQ_BAND]
    idx = np.random.default_rng(seed).choice(len(band), size=sample_size, replace=False)
    sample = band[idx] - _vocab_mean(model)
    _, _, vt = np.linalg.svd(sample, full_matrices=False)  # top-k right singular vectors = top-k PCA components
    return vt[:k]


def _pca_components(model, k):
    key = (id(model), k, PCA_SAMPLE_SIZE, PCA_SEED)
    if key not in _PCA_CACHE:
        _PCA_CACHE[key] = _fit_pca_components(model, k)
    return _PCA_CACHE[key]


def check_pca_stability(model, k=3):
    """Refit on two seeds and print |cos| between corresponding components.
    Should be >0.99; rerun manually if PCA_SAMPLE_SIZE/FREQ_BAND change."""
    a, b = _fit_pca_components(model, k, seed=1), _fit_pca_components(model, k, seed=2)
    for i in range(k):
        print(f"component {i}: |cos| = {abs(np.dot(a[i], b[i])):.4f}")


def all_but_top_k(vectors, model, k=3):
    """Mean-center, then remove the projection onto the vocabulary's top-k
    principal components (Mu & Viswanath's "all-but-the-top")."""
    centered = mean_center(vectors, model)
    components = _pca_components(model, k)
    return centered - (centered @ components.T) @ components


# --- similarity ----------------------------------------------------------------
# (vectors, words, model) -> 16x16 matrix. cosine_similarity_matrix ignores
# words/model; csls_similarity_matrix needs both (words to key the r(w)
# cache, model for cache misses). Independent alternatives, not composed.

def cosine_similarity_matrix(vectors, words=None, model=None):
    """16x16 cosine similarity. Zero-norm (OOV) vectors give similarity 0."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return unit @ unit.T


_R_CACHE_FILE = DATA_DIR / "csls_cache.json"
_r_cache = None


def _load_r_cache():
    global _r_cache
    if _r_cache is None:
        if _R_CACHE_FILE.exists():
            with open(_R_CACHE_FILE) as f:
                _r_cache = json.load(f)
        else:
            _r_cache = {}
    return _r_cache


def _save_r_cache():
    with open(_R_CACHE_FILE, "w") as f:
        json.dump(_r_cache, f)


def _mean_neighbor_similarity(vector, model):
    """r(w): mean cosine similarity between vector and its CSLS_K nearest
    neighbors in the top-FREQ_BAND frequency pool."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0.0
    pool = model.vectors[:FREQ_BAND]
    pool_norms = np.linalg.norm(pool, axis=1)
    sims = np.divide(pool @ vector, pool_norms * norm, out=np.zeros(len(pool)), where=pool_norms != 0)
    top_k = np.partition(sims, -CSLS_K)[-CSLS_K:]
    return float(top_k.mean())


def _r(word, vector, model, cache):
    """r(w) for one puzzle word, from the persisted cache or computed fresh."""
    if np.linalg.norm(vector) == 0:  # OOV
        return 0.0
    key = f"{MODEL.__name__}:{word}:{CSLS_K}:{FREQ_BAND}"
    if key not in cache:
        cache[key] = _mean_neighbor_similarity(vector, model)
    return cache[key]


def csls_similarity_matrix(vectors, words, model):
    """16x16 CSLS similarity: csls(a,b) = 2*cos(a,b) - r(a) - r(b).

    r(w) (mean similarity to w's CSLS_K nearest neighbors in the vocabulary)
    is expensive to compute — neighbor search over FREQ_BAND vectors — so it's
    cached per word in data/csls_cache.json across puzzles and runs, saved
    once per puzzle rather than once per word. Keyed by model too, so
    switching MODEL can't reuse r(w) values computed in a different vector space.
    """
    cache = _load_r_cache()
    before = len(cache)
    cos = cosine_similarity_matrix(vectors)
    r = np.array([_r(w, v, model, cache) for w, v in zip(words, vectors)])
    if len(cache) > before:
        _save_r_cache()
    return 2 * cos - r[:, None] - r[None, :]


# --- objectives ------------------------------------------------------------
# (data, group) -> float, scoring one group of 4 in isolation. sum/min_pairwise
# take a similarity matrix; centroid_score takes raw vectors (see search()).

def sum_pairwise_score(sim, group):
    return sim[np.ix_(group, group)][np.triu_indices(len(group), k=1)].sum()


def min_pairwise_score(sim, group):
    return sim[np.ix_(group, group)][np.triu_indices(len(group), k=1)].min()


def centroid_score(vectors, group):
    """Mean cosine similarity of the group's members to their own centroid."""
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


# --- model -------------------------------------------------------------------
# () -> KeyedVectors. Each downloads and caches on first use. FREQ_BAND above
# assumes vectors are frequency-sorted, true of all three of these sources.

def load_word2vec():
    import kagglehub
    from gensim.models import KeyedVectors

    path = Path(kagglehub.dataset_download("leadbest/googlenewsvectorsnegative300"))
    bin_path = next(path.rglob("*.bin"))
    return KeyedVectors.load_word2vec_format(str(bin_path), binary=True)


def load_fasttext():
    import gensim.downloader
    return gensim.downloader.load("fasttext-wiki-news-subwords-300")


def load_glove():
    import gensim.downloader
    return gensim.downloader.load("glove-wiki-gigaword-300")

def load_glove_100():
    import gensim.downloader
    return gensim.downloader.load("glove-wiki-gigaword-100")

def load_glove_200():
    import gensim.downloader
    return gensim.downloader.load("glove-wiki-gigaword-200")

def load_glove_twitter():
    import gensim.downloader
    return gensim.downloader.load("glove-twitter-200")


def load_glove_840b():
    """GloVe 840B.300d, from the local zip in data/ (not auto-downloadable via
    gensim). Extracts the text vectors once (~5.6GB) and caches them
    alongside the zip; subsequent runs load straight from the extracted file.

    Parsed by hand rather than via gensim's load_word2vec_format(no_header=True):
    that file is known to contain a handful of tokens that are themselves
    whitespace (e.g. a literal ". . ." entry), and gensim's parser does a plain
    line.split(" ") which misaligns on those lines. rsplit(" ", 300) is immune
    to that regardless of what's in the token, since it always takes the last
    300 fields as the vector. (Verified on this copy of the file: all 2,196,017
    lines already have exactly 301 space-separated fields, so this and gensim's
    parser agree here — but rsplit costs nothing and doesn't depend on that
    holding for other copies/re-downloads of the file.)
    """
    import numpy as np
    import zipfile
    from gensim.models import KeyedVectors

    zip_path = DATA_DIR / "glove.840B.300d.zip"
    txt_path = DATA_DIR / "glove.840B.300d.txt"
    if not txt_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extract("glove.840B.300d.txt", path=DATA_DIR)

    dim = 300
    kv = KeyedVectors(vector_size=dim)
    words = []
    vectors = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", dim)
            words.append(parts[0])
            vectors.append(np.asarray(parts[1:], dtype=np.float32))
    kv.add_vectors(words, np.array(vectors, dtype=np.float32))
    return kv


# --- active configuration: change these lines to swap behavior ---------------
MODEL = load_glove                    # load_word2vec | load_fasttext | load_glove | load_glove_840b
PREPROCESS = raw                       # raw | l2_normalize | mean_center | all_but_top_k
SIMILARITY = cosine_similarity_matrix  # cosine_similarity_matrix | csls_similarity_matrix
OBJECTIVE = sum_pairwise_score         # sum_pairwise_score | min_pairwise_score | centroid_score

CONFIG = {
    "embedding": MODEL.__name__,
    "lookup": "three-case fallback; phrase = underscore token else content-word average",
    "preprocess": PREPROCESS.__name__,
    "pca_sample_size": PCA_SAMPLE_SIZE,
    "pca_seed": PCA_SEED,
    "freq_band": FREQ_BAND,
    "similarity": SIMILARITY.__name__,
    "csls_k": CSLS_K,
    "objective": OBJECTIVE.__name__,
    "search": "exact brute-force over all 2,627,625 partitions",
}


# --- search --------------------------------------------------------------------

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


# Partition structure is the same for every puzzle, so build it once. GROUPS:
# the 1,820 possible 4-subsets. PARTITION_IDX: (2,627,625 x 4) indices into
# GROUPS, one row per partition, in canonical order — so the first argmax is
# already the lexicographically smallest partition (deterministic tie-break).
GROUPS = list(itertools.combinations(range(16), 4))
_GROUP_ID = {g: i for i, g in enumerate(GROUPS)}
PARTITION_IDX = np.array(
    [[_GROUP_ID[tuple(g)] for g in p] for p in _partitions(list(range(16)))],
    dtype=np.int32,
)


def search(data, group_score_fn):
    """Exact argmax partition, as a tuple of 4 sorted-index tuples.

    Requires a group-additive objective — a group's score can't depend on the
    other 3 groups in its partition. Score all 1,820 groups once, then get all
    2.6M partition scores via one vectorized index-and-sum.
    """
    group_score = np.array([group_score_fn(data, g) for g in GROUPS])
    best = np.argmax(group_score[PARTITION_IDX].sum(axis=1))
    partition = tuple(GROUPS[i] for i in PARTITION_IDX[best])
    return partition, float(group_score[PARTITION_IDX[best]].sum())


# --- solver ----------------------------------------------------------------

_KV = None


def _load_model():
    global _KV
    if _KV is None:
        _KV = MODEL()
    return _KV


def solve_detailed(words):
    """Solve one puzzle and return the full per-puzzle record from the spec."""
    kv = _load_model()
    start = time.perf_counter()

    vectors, tiers = zip(*(embed(kv, w) for w in words))
    vectors = np.array(vectors, dtype=np.float32)
    oov = [w for w, t in zip(words, tiers) if t == "oov"]

    vectors = PREPROCESS(vectors, kv)
    data = vectors if OBJECTIVE is centroid_score else SIMILARITY(vectors, words, kv)
    partition, best_score = search(data, OBJECTIVE)
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
