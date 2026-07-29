# Runs 6 (preprocess, similarity, objective) configs on glove-wiki-gigaword-300
# in parallel, each in an isolated copy of embedding.py so they don't fight
# over MODEL/PREPROCESS/SIMILARITY/OBJECTIVE globals. Prints progress as each
# config finishes, then appends all 6 results to results/result.md in order.
#
# Run and watch:
#   venv/bin/python experiments/run_glove_sweep.py
#
# Naming: each subprocess gets SWEEP_TAG=sweepN in its environment.
# embedding.py uses that (see SWEEP_TAG in embedding.py) to suffix its output
# names (embedding-sweepN, csls_cache-sweepN.json) so 6 runs finishing in the
# same second don't clobber each other's results/*.json or the shared CSLS
# r(w) cache. Unset SWEEP_TAG (i.e. running embedding.py normally) is
# unaffected — same filenames as always.

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
EMB_FILE = ROOT / "experiments" / "embedding.py"
RESULT_MD = ROOT / "results" / "result.md"
PYTHON = ROOT / "venv" / "bin" / "python3"

CONFIGS = [
    ("raw", "cosine_similarity_matrix", "min_pairwise_score"),
    ("raw", "cosine_similarity_matrix", "centroid_score"),
    ("mean_center", "cosine_similarity_matrix", "sum_pairwise_score"),
    ("all_but_top_k", "cosine_similarity_matrix", "sum_pairwise_score"),
    ("raw", "csls_similarity_matrix", "sum_pairwise_score"),
    ("all_but_top_k", "csls_similarity_matrix", "sum_pairwise_score"),
]

LINE_RE = {
    "PREPROCESS": re.compile(r"^PREPROCESS = \w+"),
    "SIMILARITY": re.compile(r"^SIMILARITY = \w+"),
    "OBJECTIVE": re.compile(r"^OBJECTIVE = \w+"),
}


def set_config(lines, preprocess, similarity, objective):
    out = []
    for line in lines:
        if LINE_RE["PREPROCESS"].match(line):
            comment = line.split("#", 1)[1].strip() if "#" in line else ""
            out.append(f"PREPROCESS = {preprocess}  # {comment}\n")
        elif LINE_RE["SIMILARITY"].match(line):
            comment = line.split("#", 1)[1].strip() if "#" in line else ""
            out.append(f"SIMILARITY = {similarity}  # {comment}\n")
        elif LINE_RE["OBJECTIVE"].match(line):
            comment = line.split("#", 1)[1].strip() if "#" in line else ""
            out.append(f"OBJECTIVE = {objective}  # {comment}\n")
        else:
            out.append(line)
    return out


def main():
    original = EMB_FILE.read_text().splitlines(keepends=True)
    if "MODEL = load_glove\n" not in "".join(original) and not any(
        l.startswith("MODEL = load_glove ") or l.startswith("MODEL = load_glove#") or l.rstrip() == "MODEL = load_glove"
        for l in original
    ):
        print("WARNING: embedding.py's MODEL is not load_glove — sweep configs assume "
              "glove-wiki-gigaword-300 (per the request). Edit MODEL back to load_glove "
              "before running, or the sweep will run against whatever model is currently set.",
              file=sys.stderr)

    import os

    workdirs = []
    procs = []

    for i, (preprocess, similarity, objective) in enumerate(CONFIGS, 1):
        workdir = Path(tempfile.mkdtemp(prefix=f"sweep_cfg{i}_"))
        exp_dir = workdir / "experiments"
        exp_dir.mkdir()

        new_lines = set_config(original, preprocess, similarity, objective)
        (exp_dir / "embedding.py").write_text("".join(new_lines))

        for name in ("connections.py", "miss_distance.py", "data", "results"):
            (workdir / name).symlink_to(ROOT / name)

        log_path = workdir / "run.log"
        env = dict(os.environ, SWEEP_TAG=f"sweep{i}")
        proc = subprocess.Popen(
            [str(PYTHON), str(exp_dir / "embedding.py")],
            cwd=str(workdir),
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs.append(proc)
        workdirs.append((i, preprocess, similarity, objective, workdir, log_path))
        print(f"[launched] config {i}: preprocess={preprocess} similarity={similarity} "
              f"objective={objective} pid={proc.pid}", flush=True)

    print(f"\nWaiting for all {len(CONFIGS)} configs to finish "
          f"(this runs an exact brute-force search over 1,134 puzzles per config)...\n", flush=True)

    remaining = {i for i, *_ in workdirs}
    while remaining:
        for i, preprocess, similarity, objective, workdir, log_path in workdirs:
            if i not in remaining:
                continue
            rc = procs[i - 1].poll()
            if rc is not None:
                remaining.discard(i)
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                print(f"[finished] config {i} ({preprocess}, {similarity}, {objective}): {status}", flush=True)
        if remaining:
            import time
            time.sleep(5)

    failures = [i for i, *_ in workdirs if procs[i - 1].returncode != 0]
    if failures:
        print(f"\nConfigs failed: {failures}. Not writing result.md.\n", file=sys.stderr)
        for i, preprocess, similarity, objective, workdir, log_path in workdirs:
            if i in failures:
                print(f"--- config {i} log ({log_path}) ---", file=sys.stderr)
                print(log_path.read_text(), file=sys.stderr)
        raise SystemExit(1)

    with open(RESULT_MD, "a") as f:
        for i, preprocess, similarity, objective, workdir, log_path in workdirs:
            config_block = (
                "CONFIG = {\n"
                '    "embedding": "glove-wiki-gigaword-300",\n'
                f'    "preprocess": "{preprocess}",\n'
                f'    "similarity": "{"csls" if "csls" in similarity else "cosine"}",\n'
                f'    "objective": "{objective}",\n'
                '    "search": "exact brute-force over all 2,627,625 partitions",\n'
                "}\n\n"
            )
            f.write(config_block)
            f.write(log_path.read_text())
            f.write("\n-------------------------------------------------------------------\n\n")

    print(f"\nAppended all {len(CONFIGS)} configs to {RESULT_MD}", flush=True)

    for _, _, _, _, workdir, _ in workdirs:
        shutil.rmtree(workdir, ignore_errors=True)
    for i in range(1, len(CONFIGS) + 1):
        tagged_cache = ROOT / "data" / f"csls_cache-sweep{i}.json"
        if tagged_cache.exists():
            tagged_cache.unlink()


if __name__ == "__main__":
    main()
