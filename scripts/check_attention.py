# Report whether the configured model's attention is causal (decoder backbone) or
# bidirectional (encoder), plus its pooling mode, and write both into run metadata.
#
# Why this matters for shared_context: Qwen3-Embedding sits on a decoder backbone.
# If attention is causal, entry 1 cannot attend to entries 2-16 in a single pass —
# each entry only sees what precedes it, which is not the symmetric contextualization
# the format is meant to test. Permutation averaging (n_permutations) only partly
# compensates. If this comes back causal, expect all-mpnet-base-v2 (a bidirectional
# encoder) to give the cleaner test in Phase 2 despite being the weaker model.
#
#   python scripts/check_attention.py --preset qwen_shared_context
#
# Loads the model config (and the SentenceTransformer for the pooling mode). Does
# not run the solver.

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "experiments"))
from sentence_embedding import PRESETS, RESULTS_DIR

# Decoder-only backbones whose self-attention is causal. Extend as new model
# families show up; the is_decoder flag below catches most that set it.
CAUSAL_TYPES = {"qwen", "qwen2", "qwen3", "llama", "mistral", "gpt2",
                "gpt_neox", "phi", "phi3", "gemma", "gemma2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="qwen_shared_context")
    args = ap.parse_args()
    model_id = PRESETS[args.preset].embedding

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "?")
    architectures = getattr(cfg, "architectures", None) or []
    # is_decoder is unreliable (Qwen3-Embedding leaves it False despite being a
    # causal decoder), so also trust the model family and a *ForCausalLM architecture.
    is_causal = (bool(getattr(cfg, "is_decoder", False))
                 or model_type in CAUSAL_TYPES
                 or any("CausalLM" in a for a in architectures))
    attention = "causal" if is_causal else "bidirectional"

    pooling = None
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_id, trust_remote_code=True)
        for module in st.modules():
            if module.__class__.__name__ == "Pooling":
                pooling = module.get_pooling_mode_str()
                break
    except Exception as e:  # pooling is informational; don't fail the check over it
        pooling = f"unknown ({type(e).__name__}: {e})"

    print(f"preset:        {args.preset}")
    print(f"model:         {model_id}")
    print(f"model_type:    {model_type}")
    print(f"architectures: {architectures}")
    print(f"attention:     {attention}")
    print(f"pooling:       {pooling}")
    if is_causal:
        print("\nNOTE: causal backbone. In shared_context each entry attends only to entries "
              "before it in the sequence; permutation averaging compensates only partly. A "
              "bidirectional encoder (e.g. all-mpnet-base-v2) is the cleaner test of this format.")

    meta = {"preset": args.preset, "model": model_id, "model_type": model_type,
            "architectures": architectures, "attention": attention, "pooling": pooling}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"model_meta-{model_id.replace('/', '_')}.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
