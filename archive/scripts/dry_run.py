# Plumbing check before spending time on all 1,134 puzzles: for one config and N
# puzzles (default 2), print the resolved input strings, encoded vector shapes, the
# 16x16 similarity summary, and the top-5 partitions with scores. Encodes only those
# puzzles and bypasses the disk cache (so it can't poison the full-run cache, which
# is keyed by embedding/format/seed regardless of how many puzzles were encoded).
#
#   python scripts/dry_run.py --preset qwen_bare --preprocess all_but_top_k
#   python scripts/dry_run.py --preset qwen_shared_context
#
# No full sweep, no caching, no results file.

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent / "experiments"))
from sentence_embedding import (GROUPS, INPUT_FORMATS, OBJECTIVES, PARTITION_IDX,
                                PREPROCESSORS, PRESETS, SIMILARITIES,
                                _encode_pooled, _encode_shared_context,
                                _resolved_inputs, load_shuffled_puzzles,
                                similarity_spread)

# The fixed downstream stack for this phase; overridable for spot checks.
DEFAULTS = {"preprocess": "all_but_top_k", "similarity": "cosine", "objective": "sum_pairwise"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="qwen_bare")
    ap.add_argument("--input-format", choices=sorted(INPUT_FORMATS))
    ap.add_argument("--preprocess", choices=sorted(PREPROCESSORS), default=DEFAULTS["preprocess"])
    ap.add_argument("--similarity", choices=sorted(SIMILARITIES), default=DEFAULTS["similarity"])
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default=DEFAULTS["objective"])
    ap.add_argument("--n", type=int, default=2, help="number of puzzles")
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k in {"input_format", "preprocess", "similarity", "objective"}}
    config = replace(PRESETS[args.preset], **overrides)
    print(f"config: {config}\n")

    puzzles = load_shuffled_puzzles(config.shuffle_seed)[:args.n]
    resolved = _resolved_inputs(puzzles, config)
    encoder = _encode_shared_context if config.input_format == "shared_context" else _encode_pooled
    vectors = encoder(puzzles, config)          # no cache read or write
    pool = vectors.reshape(-1, vectors.shape[-1])

    for i, p in enumerate(puzzles):
        print(f"=== puzzle {p['game_id']} ({p['date']}) ===")
        print("resolved inputs:")
        r = resolved[i]
        for line in (r if isinstance(r, list) else [r]):
            print(f"  {line!r}")

        v = PREPROCESSORS[config.preprocess](vectors[i], pool)
        print(f"vectors: {vectors[i].shape} -> after {config.preprocess}: {v.shape}")

        sim = SIMILARITIES[config.similarity](v, pool)
        mean, std = similarity_spread(sim)
        off = sim[np.triu_indices(16, k=1)]
        print(f"similarity 16x16: mean {mean:.3f} std {std:.3f} "
              f"min {off.min():.3f} max {off.max():.3f}")

        data = v if config.objective == "centroid" else sim
        group_score = np.array([OBJECTIVES[config.objective](data, g) for g in GROUPS])
        partition_score = group_score[PARTITION_IDX].sum(axis=1)
        top5 = np.argsort(partition_score)[::-1][:5]
        print("top-5 partitions (score : groups):")
        for rank, idx in enumerate(top5, 1):
            groups = [[p["words"][j] for j in GROUPS[g]] for g in PARTITION_IDX[idx]]
            print(f"  {rank}. {partition_score[idx]:.3f}  {groups}")
        print()


if __name__ == "__main__":
    main()
