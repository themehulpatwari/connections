# LLM solver for the Connections benchmark. One shot per puzzle, no feedback, no
# retries on wrong answers — mirrors the embedding tier's exact-search evaluation so
# the numbers stay comparable. Reuses connections.evaluate for the summary block.
#
#   venv/bin/python experiments/llm.py --preset HAIKU --slice recent100   # smoke test
#   venv/bin/python experiments/llm.py --preset HAIKU
#   venv/bin/python experiments/llm.py --preset SONNET
#
# Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Responses are cached on
# disk (results/llm_cache.jsonl), so reruns cost nothing.

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from connections import (COLORS, RESULTS_DIR, evaluate, evaluate_by_cutoff,
                         evaluate_recent, load_puzzles, solved_colors)
from miss_distance import miss_distance

# ANTHROPIC_API_KEY from repo-root .env (gitignored); the SDK reads it from the env.
load_dotenv(Path(__file__).parent.parent / ".env")

CACHE_FILE = RESULTS_DIR / "llm_cache.jsonl"      # completed results (never re-requested)
BATCH_FILE = RESULTS_DIR / "llm_batches.jsonl"    # in-flight batches (never re-submitted)
MAX_BATCH_ATTEMPTS = 2  # transient batch failures retried this many times, then scored invalid

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-4-6"

# Thinking-ON shape is adaptive+effort (not budget_tokens) for these models; Haiku 4.5
# uses budget_tokens instead.
ADAPTIVE_THINKING = {SONNET}
# Models whose thinking is ON by default when `thinking` is omitted — so "no reasoning"
# must be requested with an explicit {type:"disabled"}. Only Sonnet 5 did this; Sonnet 4.6
# and Haiku 4.5 are off-by-default, so this is empty (omitting thinking = no thinking).
ADAPTIVE_ON_BY_DEFAULT = set()
# Models that accept an explicit `temperature`. Sonnet 4.6 and Haiku 4.5 both do; Sonnet 5
# rejected any non-default temperature (400).
TEMPERATURE_OK = {HAIKU, SONNET}

# Training-cutoff dates for the days_after_cutoff metric — the reason this tier exists.
# Operator-supplied and verified (YYYY-MM-DD). A model that isn't in here yet logs
# days_after_cutoff as null rather than a fabricated number.
MODEL_CUTOFFS = {
    HAIKU: "2025-02-28",
    SONNET: "2026-02-28",
}

# (input, output) USD per 1M tokens — one entry per model this tier can run.
PRICES = {
    HAIKU: (1.00, 5.00),    # claude-haiku-4-5
    SONNET: (3.00, 15.00),  # claude-sonnet-4-6
}


def price(model):
    """Rates for `model`, or a KeyError naming it — an unpriced model must stop the run
    rather than silently report a wrong cost total."""
    if model not in PRICES:
        raise KeyError(f"no price for {model!r}: add its (input, output) USD per 1M tokens "
                       f"to PRICES before reporting a cost")
    return PRICES[model]


# --- config ------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Every lever for one run, so an experiment is a one-field edit."""

    model: str
    thinking: bool = False
    thinking_budget: int = 0        # budget_tokens on Haiku; ignored on adaptive models
    effort: str = "medium"            # adaptive-thinking models only (low|medium|high|xhigh|max)
    scratchpad: bool = False        # in-band reasoning field in the JSON
    temperature: float = 0.0        # honored only on TEMPERATURE_OK models
    max_tokens: int = 1024
    max_parse_retries: int = 2      # unparseable output only
    concurrency: int = 8
    slice: str = "all"              # "all" | "recent100"


# Three reasoning modes per model. For Sonnet 5 "no thinking" is NOT the default — the
# base preset explicitly disables adaptive thinking (see request_params); for Haiku
# thinking is off unless requested. reasoning_mode() prints the resolved mode at runtime.
PRESETS = {
    # no thinking (adaptive OFF on Sonnet). 1024 still truncated 7% of Sonnet's answers
    # (long category strings), so give the answer-only JSON room it can't plausibly need.
    "HAIKU": LLMConfig(model=HAIKU, max_tokens=2048),
    "SONNET": LLMConfig(model=SONNET, max_tokens=2048),
    # no thinking + in-band scratchpad reasoning field
    # 2048 truncated 43% of Haiku's scratchpads mid-JSON (valid runs sat at p50 1147, p99
    # 1974 — the cap, not the model, set the length). 4096 leaves real headroom.
    "HAIKU_SCRATCH": LLMConfig(model=HAIKU, scratchpad=True, max_tokens=4096),
    "SONNET_SCRATCH": LLMConfig(model=SONNET, scratchpad=True, max_tokens=4096),
    # thinking ON: Haiku needs budget_tokens < max_tokens; Sonnet uses adaptive+effort
    # CAVEAT: Haiku 4.5 does NOT honor budget_tokens as a hard cap — it thinks as long as it
    # wants and some puzzles run away past any max_tokens (truncate to all-thinking, scored
    # invalid). 6144 catches most; a few % still truncate. Prefer SONNET_THINK (adaptive
    # thinking self-limits) for the thinking tier.
    "HAIKU_THINK": LLMConfig(model=HAIKU, thinking=True, thinking_budget=1024, max_tokens=6144),
    # No config bounds Sonnet's thinking on hard puzzles: neither effort nor budget_tokens
    # is honored — some puzzles reason 8k-16k+ tokens and a few exceed any cap. 16384 lets
    # most finish (finishers stay cheap, ~1k tokens; bombs cost ~$0.25); a few still truncate.
    # Inherently expensive + variable at $15/1M output — run on a subset, not full 1134.
    "SONNET_THINK": LLMConfig(model=SONNET, thinking=True, effort="low", max_tokens=16384),
}


def reasoning_mode(config):
    """Human-readable resolved reasoning mode, printed at runtime so it's unambiguous
    which of {off, scratchpad, thinking} a run is."""
    if config.thinking:
        if config.model in ADAPTIVE_THINKING:
            return f"adaptive thinking (effort={config.effort})"
        return f"extended thinking (budget_tokens={config.thinking_budget})"
    off = "thinking OFF (adaptive disabled)" if config.model in ADAPTIVE_ON_BY_DEFAULT else "thinking OFF"
    return off + (" + in-band scratchpad" if config.scratchpad else "")


# --- frozen prompt (do not alter across models) ------------------------------

SYSTEM_HEAD = """You are solving a New York Times Connections puzzle.

You are given 16 words. Partition them into exactly 4 groups of 4. Each group
shares one specific connection.

Structure:
- The categories vary in difficulty. Most are semantic, but a category may
  instead be based on wordplay: homophones, hidden words, anagrams, words that
  all precede or follow a common word, or words that become new words when a
  letter is added or removed. Check whether any such category is present. Often
  none is — do not force one.
- The puzzle is built around red herrings. Several words will plausibly fit a
  category they do not belong to. Exactly one partition of all 16 works.

Method:
- Do not accept a group just because 4 words fit. Check whether a 5th word also
  fits. If one does, the category is wrong or its boundary is elsewhere.
- Start from whichever category is most constrained — the one where the 4 words
  have the fewest alternative homes.
- Every word is used exactly once.
"""
# No format instruction: the grammar guarantees schema-valid JSON with no preamble, and
# telling the model to emit JSON only pushed it toward minimal output.
SYSTEM_TAIL = """
Give each group a specific category label and a confidence from 0.0 to 1.0.
Every one of the 16 words appears in exactly one group — never repeat a word.
"""
# Identical for every preset — the scratchpad field is part of the schema, not the prompt.
SYSTEM_PROMPT = SYSTEM_HEAD + SYSTEM_TAIL


def user_message(words):
    """The 16 words, uppercase, comma-separated, in dataset order. No shuffling."""
    return ", ".join(w.upper() for w in words)


# --- request shaping (model-aware) -------------------------------------------

# Bump when request-building semantics change so the on-disk cache invalidates itself
# (the prompt sha only covers the prompt text, not output_config / thinking shape).
REQUEST_VERSION = 3


def output_schema(config, board_words):
    """JSON schema for structured outputs. A prompt instruction alone doesn't stop Haiku
    from emitting chain-of-thought prose instead of JSON; output_config.format makes the
    API guarantee schema-valid JSON with no preamble.

    Arity is encoded as required properties (group_1..group_4, word_1..word_4) because the
    API rejects array-length constraints, and the word enum is this board's 16 words — so
    an empty answer, a wrong group/word count, and an invented word are all ungrammatical.
    Only cross-field uniqueness is left for validate() to catch."""
    word = {"type": "string", "enum": list(board_words)}
    group = {
        "type": "object",
        "properties": {
            "category": {"type": "string"},
            "word_1": word, "word_2": word, "word_3": word, "word_4": word,
            "confidence": {"type": "number"},
        },
        "required": ["category", "word_1", "word_2", "word_3", "word_4", "confidence"],
        "additionalProperties": False,
    }
    props = {f"group_{i}": group for i in range(1, 5)}
    required = [f"group_{i}" for i in range(1, 5)]
    # reasoning first so the model reasons before committing groups. Verified against the
    # cached HAIKU_SCRATCH responses: "reasoning" precedes "group" in all 664 that carry
    # both keys (the rest truncated mid-reasoning), so the grammar does emit schema order.
    if config.scratchpad:
        props = {"reasoning": {"type": "string"}, **props}
        required = ["reasoning"] + required
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


def request_params(config, system, user, board_words):
    params = {
        "model": config.model,
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "format": {"type": "json_schema", "schema": output_schema(config, board_words)},
        },
    }
    # Extended thinking forbids a non-default temperature (400), so only send it when
    # thinking is off. Sonnet 5 would also reject it, hence TEMPERATURE_OK.
    if config.model in TEMPERATURE_OK and not config.thinking:
        params["temperature"] = config.temperature
    if config.thinking:
        if config.model in ADAPTIVE_THINKING:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"]["effort"] = config.effort
        else:
            params["thinking"] = {"type": "enabled", "budget_tokens": config.thinking_budget}
    elif config.model in ADAPTIVE_ON_BY_DEFAULT:
        # Thinking is on by default here — turn it off explicitly.
        params["thinking"] = {"type": "disabled"}
    return params


# --- parsing / validation ----------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_groups(text):
    """Strip fences (defensively), json.loads, and pull group_1..group_4 out as four lists
    of four uppercased words, alongside their confidences. Returns (groups, confidences),
    or (None, None) if the output isn't structurally parseable (the only case we retry).
    One parse serves both — the caller never re-reads the same text."""
    stripped = _FENCE.sub("", text.strip())
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    groups, confidences = [], []
    for i in range(1, 5):
        g = data.get(f"group_{i}")
        if not isinstance(g, dict):
            return None, None
        words = [g.get(f"word_{j}") for j in range(1, 5)]
        if any(not isinstance(w, str) for w in words):
            return None, None
        groups.append([w.strip().upper() for w in words])
        conf = g.get("confidence")
        confidences.append(conf if isinstance(conf, (int, float)) else None)
    return groups, confidences


_NONWORD = re.compile(r"[^A-Z0-9]")


def _key(word):
    """Match words on letters/digits only. Boards contain spaces, hyphens and periods
    (ICE CUBE, Q-TIP, J.LO); a model that writes ICECUBE or Q TIP picked the right word
    and must not be scored invalid for the punctuation."""
    return _NONWORD.sub("", word.upper())


def committed_groups(text, puzzle_words):
    """The four groups the model actually committed to, in board casing — regardless of
    whether they form a legal partition. This is what a near-miss deserves to be scored
    against: an answer that names three true groups and slips one word is not a zero.

    The grammar fixes the answer at four groups of four, so there is nothing to select
    between. Returns ([[]], []) when nothing parseable came back, so the miss-distance
    metric reads "missed everything" rather than crashing on an empty guess."""
    groups, confidences = parse_groups(text)
    if groups is None:
        return [[]], []
    key_to_orig = {_key(w): w for w in puzzle_words}
    return [[key_to_orig.get(_key(w), w) for w in g] for g in groups], confidences


def no_json_reason(text, stop_reason=None):
    """Why nothing parseable came back. Structured outputs guarantee well-formed JSON, so a
    parse failure means the response was cut off — at max_tokens if the API says so, and
    with no text at all when the whole budget went to thinking. stop_reason is the
    authority; the text-emptiness guess is only for legacy records that never stored one."""
    if stop_reason is not None:
        return "truncated" if stop_reason == "max_tokens" else "unparseable"
    return "truncated" if text.strip() else "unparseable"


def validate(raw, puzzle_words):
    """A parseable partition is valid iff it is 4 groups of 4 whose multiset equals the
    puzzle's 16 words (compared on letters/digits only). Returns (True, groups-in-
    original-case, "ok") or (False, None, reason). We never repair an invalid partition;
    the reason exists so the harness can tell a model failure from a harness failure (we
    cut it off mid-JSON).

    Under the current grammar the shape checks can't fire and no word can be invented, so
    not_a_partition means exactly one thing: the model used a word twice."""
    if len(raw) != 4 or any(len(g) != 4 for g in raw):
        return False, None, "empty" if not raw else "bad_shape"
    key_to_orig = {_key(w): w for w in puzzle_words}
    flat = [_key(w) for g in raw for w in g]
    if sorted(flat) != sorted(key_to_orig):
        return False, None, "not_a_partition"
    return True, [[key_to_orig[_key(w)] for w in g] for g in raw], "ok"


# --- one puzzle --------------------------------------------------------------

def cache_key(config, sha, puzzle_id):
    """max_tokens is deliberately NOT part of the key: raising it must not invalidate the
    whole cache. Use --redo-truncated to re-request only the responses a smaller budget cut
    off, and --rescore to re-apply scorer changes to what's already collected."""
    return json.dumps([REQUEST_VERSION, config.model, config.thinking, config.thinking_budget,
                       config.effort, config.scratchpad, config.temperature, sha, puzzle_id])


def days_after_cutoff(model, puzzle_date):
    try:
        cutoff = date.fromisoformat(MODEL_CUTOFFS[model])
    except (KeyError, TypeError, ValueError):
        return None  # placeholder / unfilled cutoff
    return (date.fromisoformat(puzzle_date) - cutoff).days


def _record(config, puzzle, predicted, invalid, retries, raw, in_tok, out_tok, latency, sha,
            batch=False, reason="ok", stop_reason=None, retry_tok=(0, 0)):
    solved = solved_colors(predicted, puzzle["answer"])
    # Second, lenient scoring of the model's OWN answer. For a valid response these are the
    # same groups; for an invalid one `predicted` is a placeholder the model never emitted,
    # so scoring only that throws away every true group a near-miss did find.
    committed, confidence = committed_groups(raw, puzzle["words"])
    solved_committed = solved_colors(committed, puzzle["answer"])
    return {
        "cache_key": cache_key(config, sha, puzzle["game_id"]),
        "puzzle_id": puzzle["game_id"],
        "puzzle_date": puzzle["date"],
        "model": config.model,
        "scratchpad": config.scratchpad,
        "thinking": config.thinking,
        "batch": batch,               # via Batches API (50% price) vs live
        "groups_correct": len(solved),
        "win": len(solved) == 4,
        "per_color_correct": {c: (c in solved) for c in COLORS},
        # null when invalid: the old value was miss distance to the placeholder partition,
        # a number about the harness's filler rather than about the model.
        "miss_distance_per_color": None if invalid else miss_distance(predicted, puzzle["answer"],
                                                                     COLORS),
        "groups_correct_committed": len(solved_committed),
        "win_committed": len(solved_committed) == 4,
        "per_color_correct_committed": {c: (c in solved_committed) for c in COLORS},
        "miss_distance_committed": miss_distance(committed, puzzle["answer"], COLORS),
        "committed_groups": committed,
        "confidence": confidence,     # the model's own per-group confidence, in group order
        "invalid": invalid,
        # why it was scored invalid: truncated | unparseable | empty | bad_shape |
        # not_a_partition | error ("ok" when valid). truncated means our max_tokens was too
        # small, not that the model failed — it is a harness fault, tracked separately.
        "invalid_reason": reason,
        "stop_reason": stop_reason,
        "parse_retries": retries,
        # the batch path can't retry a parse failure; live can. Recorded so two runs of the
        # same preset are distinguishable in the results file.
        "retry_enabled": not batch and config.max_parse_retries > 0,
        "raw_response": raw,
        # Final attempt only, so `output_tokens == max_tokens` still means "truncated".
        # Tokens burned by abandoned attempts are kept separately (and still costed).
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "retry_tokens": {"input": retry_tok[0], "output": retry_tok[1]},
        "thinking_tokens": None,  # not exposed separately by the API (billed within output)
        "latency_s": latency,
        "days_after_cutoff": days_after_cutoff(config.model, puzzle["date"]),
        "prompt_sha256": sha,
        "predicted_groups": predicted,
    }


def build_record(config, sha, puzzle, text, in_tok, out_tok, latency, retries, batch,
                 stop_reason=None, retry_tok=(0, 0)):
    """Parse -> validate -> record for one response text (shared by live and batch).
    Anything that doesn't parse to a valid 4x4 partition is scored invalid, no repair."""
    words = puzzle["words"]
    predicted, invalid = [list(words[i:i + 4]) for i in range(0, 16, 4)], True
    groups, _ = parse_groups(text)
    if groups is None:
        reason = no_json_reason(text, stop_reason)
    else:
        ok, translated, reason = validate(groups, words)
        if ok:
            predicted, invalid = translated, False
    return _record(config, puzzle, predicted, invalid, retries, text, in_tok, out_tok, latency,
                   sha, batch, reason, stop_reason, retry_tok)


async def solve_one(client, config, system, sha, puzzle, sem, cache, lock):
    key = cache_key(config, sha, puzzle["game_id"])
    if key in cache:
        return cache[key], False

    words = puzzle["words"]
    params = request_params(config, system, user_message(words), words)
    fallback = [list(words[i:i + 4]) for i in range(0, 16, 4)]  # scores ~0 for invalid

    in_tok = out_tok = latency = retries = 0
    retry_in = retry_out = 0
    predicted, invalid, raw = fallback, True, ""
    reason, stop_reason = "unparseable", None
    async with sem:
        while True:
            start = time.perf_counter()
            resp = await client.messages.create(**params)
            latency += time.perf_counter() - start
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
            stop_reason = resp.stop_reason
            raw = "".join(b.text for b in resp.content if b.type == "text")

            groups, _ = parse_groups(raw)
            if groups is None:                        # unparseable — retry
                if retries < config.max_parse_retries:
                    retries += 1
                    retry_in, retry_out = retry_in + in_tok, retry_out + out_tok
                    # A retry at identical params is the same request paid for twice.
                    # Truncation causes nearly every parse failure, so give it room; and
                    # where a temperature is legal at all, take it off 0 so a model that
                    # rambled its way past the budget can take a different path.
                    params["max_tokens"] *= 2
                    if "temperature" in params:
                        params["temperature"] = 1.0
                    continue
                break                                 # exhausted -> invalid=True, fallback
            ok, translated, reason = validate(groups, words)  # parseable-but-wrong: no retry
            if ok:
                predicted, invalid = translated, False
            break

    if invalid and parse_groups(raw)[0] is None:
        reason = no_json_reason(raw, stop_reason)
    record = _record(config, puzzle, predicted, invalid, retries, raw, in_tok, out_tok, latency,
                     sha, False, reason, stop_reason, (retry_in, retry_out))
    cache[key] = record
    async with lock:
        with CACHE_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return record, True


# --- run ---------------------------------------------------------------------

# Cached failures that are worth re-requesting once max_tokens has been raised: both mean
# "no JSON came back", which the record can't distinguish from a too-small budget. Opt-in
# via --redo-truncated because re-requesting costs money.
REDO_REASONS = {"truncated", "unparseable"}


def invalid_reason(record):
    """The stored reason, re-derived for legacy records written before the field existed."""
    if not record["invalid"]:
        return "ok"
    if record.get("error"):
        return "error"
    stored = record.get("invalid_reason")
    if stored is not None:
        return stored
    raw = record.get("raw_response") or ""
    groups, _ = parse_groups(raw)
    if groups is None:
        return no_json_reason(raw, record.get("stop_reason"))
    return "empty" if not groups else ("bad_shape" if len(groups) != 4
                                       or any(len(g) != 4 for g in groups) else "not_a_partition")


def _is_redo(record):
    """No JSON came back — worth re-requesting under a bigger budget. A model failure that
    produced a well-formed partition is never a redo candidate."""
    return not record.get("error") and invalid_reason(record) in REDO_REASONS


def load_cache(redo_truncated=False):
    cache = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["cache_key"]] = r  # append-only: the last line for a key wins
    if redo_truncated:
        for key in [k for k, r in cache.items() if _is_redo(r)]:
            del cache[key]  # a miss -> re-requested with the current max_tokens
    return cache


def rescore_cache(cache, config, sha, puzzles):
    """Rebuild cached records from their stored raw_response, in memory only. Free (no API
    calls) and how a scorer/parser fix reaches results that are already collected — without
    it, a cached record keeps whatever verdict the scorer of the day gave it."""
    changed = 0
    for puzzle in puzzles:
        key = cache_key(config, sha, puzzle["game_id"])
        old = cache.get(key)
        if old is None or old.get("error"):
            continue
        retry_tok = old.get("retry_tokens") or {}
        new = build_record(config, sha, puzzle, old["raw_response"], old["input_tokens"],
                           old["output_tokens"], old.get("latency_s"),
                           old.get("parse_retries", 0), old.get("batch", False),
                           old.get("stop_reason"),
                           (retry_tok.get("input", 0), retry_tok.get("output", 0)))
        if new["invalid"] != old["invalid"] or new["groups_correct"] != old["groups_correct"]:
            changed += 1
        cache[key] = new
    if changed:
        print(f"rescored {changed} cached record(s) under the current scorer")
    return changed


def cache_put(cache, record):
    """Add a completed record to the in-memory cache and append it to disk."""
    cache[record["cache_key"]] = record
    with CACHE_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


async def run(config, puzzles, redo_truncated=False, rescore=False):
    from anthropic import AsyncAnthropic

    system = SYSTEM_PROMPT
    sha = hashlib.sha256(system.encode()).hexdigest()
    cache = load_cache(redo_truncated)
    if rescore:
        rescore_cache(cache, config, sha, puzzles)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic(max_retries=8)  # SDK does exponential backoff on 429/5xx
    sem = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()
    results = await asyncio.gather(
        *(solve_one(client, config, system, sha, p, sem, cache, lock) for p in puzzles)
    )
    await client.close()
    records = [r for r, _ in results]
    n_called = sum(1 for _, called in results if called)
    return records, n_called


def total_cost(records):
    """List price of every attempt, retries included. Cache reads aren't modelled because
    nothing here sets cache_control — every input token bills at the base rate, by choice.
    Thinking needs no term of its own: the API bills it as output and already counts it in
    usage.output_tokens."""
    total = 0.0
    for r in records:
        in_rate, out_rate = price(r["model"])
        retry = r.get("retry_tokens") or {}
        c = ((r["input_tokens"] + retry.get("input", 0)) / 1e6 * in_rate
             + (r["output_tokens"] + retry.get("output", 0)) / 1e6 * out_rate)
        total += c * 0.5 if r.get("batch") else c  # Batches API is 50% off
    return total


# --- batch mode (async, 50% cheaper) -----------------------------------------

def config_sig(config, sha):
    """Cache-key prefix minus the puzzle id — identifies one run config, so we only
    collect/track batches that belong to the current preset+prompt."""
    return json.dumps([REQUEST_VERSION, config.model, config.thinking, config.thinking_budget,
                       config.effort, config.scratchpad, config.temperature, sha])


def load_batches():
    if not BATCH_FILE.exists():
        return []
    return [json.loads(l) for l in BATCH_FILE.read_text().splitlines() if l.strip()]


def save_batches(entries):
    BATCH_FILE.write_text("".join(json.dumps(e) + "\n" for e in entries))


def _mark_failed(cache, config, sha, puzzle, note):
    """Resolve a puzzle we can't get a result for by caching it as invalid (scored 0,
    zero tokens), so it stops blocking the run and never resubmits."""
    rec = build_record(config, sha, puzzle, "", 0, 0, None, 0, batch=True)
    rec["raw_response"], rec["error"], rec["invalid_reason"] = note, True, "error"
    cache_put(cache, rec)


def collect_batches(client, config, sha, cache, puzzles_by_id, entries):
    """Retrieve any ended batches for this config and write their results into the cache.
    succeeded -> record; permanent error (invalid_request) -> scored invalid now; transient
    error -> logged on the entry to retry (see resolve_capped). Returns (collected,
    permanent, transient). Mutates cache and entries."""
    sig = config_sig(config, sha)
    collected = permanent = transient = 0
    for entry in entries:
        if entry["status"] != "submitted" or entry["sig"] != sig:
            continue
        batch = client.messages.batches.retrieve(entry["batch_id"])
        if batch.processing_status != "ended":
            continue
        for result in client.messages.batches.results(entry["batch_id"]):
            pid = int(result.custom_id)
            if cache_key(config, sha, pid) in cache:
                continue
            if result.result.type == "succeeded":
                msg = result.result.message
                text = "".join(b.text for b in msg.content if b.type == "text")
                cache_put(cache, build_record(config, sha, puzzles_by_id[pid], text,
                          msg.usage.input_tokens, msg.usage.output_tokens, None, 0, batch=True,
                          stop_reason=msg.stop_reason))
                collected += 1
            else:
                etype = getattr(getattr(result.result, "error", None), "type", "") or ""
                if "invalid_request" in etype:  # deterministic — retrying won't help
                    _mark_failed(cache, config, sha, puzzles_by_id[pid], f"batch permanent error: {etype}")
                    permanent += 1
                else:
                    entry.setdefault("errored", []).append(pid)  # transient — eligible for retry
                    transient += 1
        entry["status"] = "done"
    save_batches(entries)
    return collected, permanent, transient


def resolve_capped(config, sha, cache, puzzles_by_id, entries):
    """Score-invalid any puzzle that has failed transiently MAX_BATCH_ATTEMPTS times, so a
    persistently-failing item can't stall the run forever. Returns how many were marked."""
    sig = config_sig(config, sha)
    attempts = Counter(pid for e in entries if e["sig"] == sig and e["status"] == "done"
                       for pid in e.get("errored", []))
    marked = 0
    for pid, n in attempts.items():
        if n >= MAX_BATCH_ATTEMPTS and cache_key(config, sha, pid) not in cache:
            _mark_failed(cache, config, sha, puzzles_by_id[pid], f"batch failed after {n} attempts")
            marked += 1
    return marked


def submit_batch(client, config, system, sha, need):
    """Create one batch for the puzzles in `need` and record it in the ledger."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [
        Request(custom_id=str(p["game_id"]),
                params=MessageCreateParamsNonStreaming(
                    **request_params(config, system, user_message(p["words"]), p["words"])))
        for p in need
    ]
    # Fail loudly rather than submit unconstrained requests: if the params type ever drops
    # output_config, batch runs would be ungrammared while live runs weren't, and the two
    # sets of numbers would not be comparable.
    missing = [r for r in requests if "output_config" not in r["params"]]
    if missing:
        raise RuntimeError(f"output_config was dropped from {len(missing)}/{len(requests)} batch "
                           f"requests — they would run unconstrained; not submitting")
    batch = client.messages.batches.create(requests=requests)
    entries = load_batches()
    entries.append({
        "batch_id": batch.id,
        "sig": config_sig(config, sha),
        "status": "submitted",
        "created": datetime.now().isoformat(timespec="seconds"),
        "custom_ids": [p["game_id"] for p in need],
    })
    save_batches(entries)
    print("\n--- batch submitted ---")
    print(f"  batch_id:   {batch.id}")
    print(f"  status:     {batch.processing_status}")
    print(f"  requests:   {len(need)}  ({batch.request_counts.processing} processing)")
    print(f"  created_at: {batch.created_at}")
    print(f"  expires_at: {batch.expires_at}")
    print(f"  tracked in: {BATCH_FILE}")
    print(f"  collect:    re-run the same command with --collect (or --batch), or add --wait")
    print("-----------------------")
    return batch.id


def _progress(records, n):
    got = len(records)
    if got:
        acc = sum(r["groups_correct"] for r in records) / (got * 4)
        print(f"progress: {got}/{n} collected  (grouping-so-far {acc:.3f} — PARTIAL, not final)")
    else:
        print(f"progress: 0/{n} collected")


def run_batch(config, puzzles, wait, submit=True, poll_s=30, redo_truncated=False,
              rescore=False):
    """Collect finished batches into the cache and (unless submit=False) submit the puzzles
    still missing. Non-blocking unless `wait` (polls every poll_s while batches are in
    flight). Persistently-failing puzzles are scored invalid after MAX_BATCH_ATTEMPTS so the
    run always terminates. Returns (records_for_solved, pending_count)."""
    from anthropic import Anthropic

    system = SYSTEM_PROMPT
    sha = hashlib.sha256(system.encode()).hexdigest()
    cache = load_cache(redo_truncated)
    puzzles_by_id = {p["game_id"]: p for p in load_puzzles()}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = Anthropic()
    sig = config_sig(config, sha)
    if rescore:
        rescore_cache(cache, config, sha, puzzles)

    def solved_and_pending():
        keys = [cache_key(config, sha, p["game_id"]) for p in puzzles]
        recs = [cache[k] for k in keys if k in cache]
        return recs, len(puzzles) - len(recs)

    def collect():
        entries = load_batches()
        c, perm, trans = collect_batches(client, config, sha, cache, puzzles_by_id, entries)
        marked = resolve_capped(config, sha, cache, puzzles_by_id, entries)
        if c or perm or trans or marked:
            print(f"collected {c}  |  errored: {perm} permanent, {trans} transient  "
                  f"|  gave up (scored invalid): {marked}")

    def in_flight():
        return any(e["status"] == "submitted" and e["sig"] == sig for e in load_batches())

    collect()
    records, pending = solved_and_pending()
    _progress(records, len(puzzles))

    if submit:
        pending_keys = {cache_key(config, sha, pid) for e in load_batches()
                        if e["status"] == "submitted" and e["sig"] == sig for pid in e["custom_ids"]}
        need = [p for p in puzzles if cache_key(config, sha, p["game_id"]) not in cache
                and cache_key(config, sha, p["game_id"]) not in pending_keys]
        if need:
            submit_batch(client, config, system, sha, need)
        records, pending = solved_and_pending()

    while wait and pending and in_flight():
        print(f"waiting on {pending} puzzles… (poll {poll_s}s)")
        time.sleep(poll_s)
        collect()
        records, pending = solved_and_pending()
        _progress(records, len(puzzles))
    return records, pending


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="HAIKU", choices=sorted(PRESETS))
    ap.add_argument("--slice", choices=["all", "recent100"])
    ap.add_argument("--batch", action="store_true",
                    help="submit via the Batches API (50%% cheaper, async); re-run to collect")
    ap.add_argument("--collect", action="store_true",
                    help="retrieve batch results only — never submits anything new")
    ap.add_argument("--wait", action="store_true", help="with --batch, poll until results are ready")
    ap.add_argument("--redo-truncated", action="store_true",
                    help="re-request cached puzzles whose response never parsed (truncated at\n"
                         "max_tokens or non-JSON) instead of reusing the scored-invalid record")
    ap.add_argument("--rescore", action="store_true",
                    help="re-parse and re-score already-cached responses under the current "
                         "scorer (free, no API calls)")
    args = ap.parse_args()

    config = PRESETS[args.preset]
    if args.slice:
        config = replace(config, slice=args.slice)

    puzzles = load_puzzles()
    if config.slice == "recent100":
        puzzles = puzzles[-100:]

    batch_mode = args.batch or args.collect
    if batch_mode:
        records, pending = run_batch(config, puzzles, wait=args.wait, submit=args.batch,
                                  redo_truncated=args.redo_truncated, rescore=args.rescore)
        if pending:
            hint = "Re-run with --batch to retry failures / collect more" + (
                "" if args.wait else " (or add --wait to poll now)")
            print(f"\n{pending}/{len(puzzles)} puzzles not yet resolved. {hint}.")
            sys.exit(0)
        n_called = None
    else:
        records, n_called = asyncio.run(run(config, puzzles, args.redo_truncated, args.rescore))

    # evaluate() re-calls solve(words); serve the already-collected answers instead. Keyed
    # on game_id, with words -> game_id built up front: two puzzles that shared a word list
    # would otherwise collapse into one entry silently, so refuse to score at all instead.
    game_ids = {}
    for p in puzzles:
        key = tuple(p["words"])
        if key in game_ids:
            sys.exit(f"puzzles {game_ids[key]} and {p['game_id']} have identical word lists; "
                     f"evaluate() can't tell them apart from words alone")
        game_ids[key] = p["game_id"]

    def served(field):
        by_id = {r["puzzle_id"]: r[field] for r in records}
        return lambda words: by_id[game_ids[tuple(words)]]

    cached_solve = served("predicted_groups")
    name = f"llm_{args.preset.lower()}"
    print(f"=== {name} ({config.model}, slice={config.slice}, {len(puzzles)} puzzles) ===")
    print(f"reasoning:          {reasoning_mode(config)}")
    # No miss-distance histogram here: an invalid answer is scored as a placeholder
    # partition, so its distances would describe the harness's filler. The credited block
    # below scores what the model actually emitted and reports the real one.
    evaluate(name, cached_solve, puzzles, show_miss_distance=False)

    # Cutoff split — the reason this tier exists: does the model do better on puzzles it
    # could have seen in training (<= cutoff) than after? Uses the model's OWN cutoff.
    # Meaningful only on --slice all; recent100 is entirely after both models' cutoffs.
    cutoff_summary = recent_summary = None
    try:
        date.fromisoformat(MODEL_CUTOFFS[config.model])
        print(f"\n=== knowledge-cutoff split ({config.model}) ===")
        cutoff_summary = evaluate_by_cutoff(cached_solve, puzzles, MODEL_CUTOFFS[config.model])
        print("\n=== recent-window (has puzzles gotten harder?) ===")
        recent_summary = evaluate_recent(cached_solve, puzzles, n=100)
    except (KeyError, TypeError, ValueError):
        print(f"\ncutoff split skipped: MODEL_CUTOFFS[{config.model}] is not a real date yet")

    # LLM-only metrics the embedding tier can't produce (omit oracle_rank — one partition,
    # not a ranking).
    invalid = sum(r["invalid"] for r in records)
    print(f"\ninvalid_rate:      {invalid / len(records):.3f}  ({invalid}/{len(records)})")
    # Break the rate down: truncated is our budget being too small (raise max_tokens and
    # re-run those); the rest are genuine model failures at this reasoning level.
    breakdown = Counter(invalid_reason(r) for r in records if r["invalid"])
    for why, n in breakdown.most_common():
        label = "  (harness: raise max_tokens)" if why == "truncated" else ""
        print(f"  {why:<16} {n / len(records):.3f}  ({n}/{len(records)}){label}")
    tail = "  (batch, 50% off)" if batch_mode else f"  ({n_called} puzzles solved this run)"
    print(f"total token cost:  ${total_cost(records):.2f}{tail}")

    # Second scoring of the same responses against the groups the model actually emitted.
    # The strict block above is the headline (it's what compares to the embedding tier, which
    # can only ever emit legal partitions); this one separates "found nothing" from "found
    # three of four groups but reused a word", which strict scoring reports identically as 0.
    print("\n=== credited-answer scoring (the model's own groups, invalid partitions included) ===")
    print("  strict scoring above replaces an invalid answer with a placeholder partition;")
    print("  here every emitted group is checked against truth — including the miss distances.")
    committed_summary = evaluate(f"{name}_committed", served("committed_groups"), puzzles)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    detail_file = RESULTS_DIR / f"{name}-detail-{stamp}.json"
    detail_file.write_text(json.dumps(
        {"config": asdict(config), "prompt_sha256": records[0]["prompt_sha256"] if records else None,
         "invalid_rate": invalid / len(records), "total_cost": total_cost(records),
         "invalid_breakdown": dict(breakdown), "committed": committed_summary,
         "cutoff": cutoff_summary, "recent": recent_summary, "puzzles": records}, indent=2))
    print(f"\nSaved per-puzzle detail to {detail_file}")
