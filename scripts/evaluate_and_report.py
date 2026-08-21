"""
Day 10 (part 2): Turn data/eval_generations.jsonl into the actual numbers
the project's headline metric needs -- an LLM-as-judge pairwise win-rate
(SFT+DPO vs. SFT-only) and per-dimension rubric scores (1-5) for all three
variants -- plus a final markdown report with qualitative examples.

Reuses (imports, doesn't duplicate) the rule-based checks, the JSON-object
extraction fix, and the provider config from label_dpo_pairs.py.

Position-bias control: which variant is shown as "A" vs "B" to the judge
is randomized per row (not always SFT=A, DPO=B), and the win is mapped
back to the actual model afterward -- otherwise a judge with any
positional preference would silently bias the win-rate.

Requires an API key, same as Day 3/8:
  GEMINI_API_KEY=...  (in .env)

Usage (run from the voiceforge/ repo root):
  python scripts/evaluate_and_report.py \\
      --generations data/eval_generations.jsonl \\
      --voice docs/voice_guidelines.md \\
      --out docs/eval_results.md
"""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from label_dpo_pairs import (  # noqa: E402
    PROVIDERS, extract_json_object, check_candidate,
)
from generate_completions import parse_voice_doc  # noqa: E402

random.seed(42)

RUBRIC_DIMENSIONS = ["tone_adherence", "vocabulary_compliance", "cta_correctness", "concrete_detail"]


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

def build_pairwise_judge_prompts(voice_doc_text):
    shared_text, persona_text = parse_voice_doc(voice_doc_text)
    prompts = {}
    for voice_key, section in persona_text.items():
        body = f"{shared_text}\n\n{section}" if section else voice_doc_text
        prompts[voice_key] = (
            "You are an expert brand-voice reviewer comparing two marketing copy "
            "completions (A and B) for the same brief, against the brand voice "
            "guide below. Judge ONLY on voice-guide adherence -- tone, vocabulary, "
            "banned phrases, emoji policy, CTA rule, required concrete detail. Do "
            "not reward generic fluency over rule adherence.\n\n"
            'Respond with ONLY a JSON object: {"winner": "A", "reason": "<one sentence>"} '
            'or {"winner": "B", "reason": "<one sentence>"} or '
            '{"winner": "tie", "reason": "<one sentence>"}.\n\n'
            f"{body}"
        )
    return prompts


def build_rubric_judge_prompts(voice_doc_text):
    shared_text, persona_text = parse_voice_doc(voice_doc_text)
    prompts = {}
    for voice_key, section in persona_text.items():
        body = f"{shared_text}\n\n{section}" if section else voice_doc_text
        prompts[voice_key] = (
            "You are an expert brand-voice reviewer. Score the completion below "
            "against the brand voice guide, on these 4 dimensions, each 1-5 "
            "(5 = fully adheres, 1 = clearly violates or absent):\n"
            "- tone_adherence: matches the tone words and sentence style\n"
            "- vocabulary_compliance: avoids banned phrases/words-to-avoid\n"
            "- cta_correctness: follows this format's CTA rule (required/forbidden/optional)\n"
            "- concrete_detail: includes a real practical detail where the format requires one\n\n"
            'Respond with ONLY a JSON object: {"tone_adherence": <1-5>, '
            '"vocabulary_compliance": <1-5>, "cta_correctness": <1-5>, '
            '"concrete_detail": <1-5>}\n\n'
            f"{body}"
        )
    return prompts


def build_pairwise_user_prompt(brief, fmt, text_a, text_b):
    return f"FORMAT: {fmt}\n\nBRIEF: {brief}\n\nCANDIDATE A: {text_a}\n\nCANDIDATE B: {text_b}\n\nJSON only."


def build_rubric_user_prompt(brief, fmt, text):
    return f"FORMAT: {fmt}\n\nBRIEF: {brief}\n\nCOMPLETION: {text}\n\nJSON only."


def parse_rubric_response(raw_text):
    data = extract_json_object(raw_text)
    scores = {}
    for dim in RUBRIC_DIMENSIONS:
        val = data.get(dim)
        if not isinstance(val, (int, float)) or not (1 <= val <= 5):
            raise ValueError(f"missing/invalid score for {dim}: {val!r}")
        scores[dim] = val
    return scores


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def call_with_retry(call_fn, system_prompt, user_prompt, model, max_tokens, sleep, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            return call_fn(system_prompt, user_prompt, model, max_tokens)
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default="data/eval_generations.jsonl")
    ap.add_argument("--voice", default="docs/voice_guidelines.md")
    ap.add_argument("--out", default="docs/eval_results.md")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--n_examples", type=int, default=6, help="qualitative examples to include in the report")
    args = ap.parse_args()

    provider = PROVIDERS[args.provider]
    if provider["key_env_var"] not in os.environ:
        sys.exit(f"Set {provider['key_env_var']} before running (see docstring).")
    call_fn = provider["call_fn"]
    model = args.model or provider["default_model"]

    rows = load_jsonl(args.generations)
    if not rows:
        sys.exit(f"No rows in {args.generations} -- did Day 10 part 1's notebook run?")
    print(f"Loaded {len(rows)} evaluated prompts")

    voice_doc_text = Path(args.voice).read_text(encoding="utf-8")
    pairwise_prompts = build_pairwise_judge_prompts(voice_doc_text)
    rubric_prompts = build_rubric_judge_prompts(voice_doc_text)

    # --- 1. rule-based compliance (already computed in the notebook; recompute
    # here too via the same imported function, so this script doesn't have to
    # trust the notebook's schema/version) ---
    rule_clean_rate = {}
    for variant in ("base", "sft", "dpo"):
        clean = sum(
            1 for r in rows
            if not check_candidate(r[variant], r["voice"], r["format"])
        )
        rule_clean_rate[variant] = clean / len(rows)

    # --- 2. pairwise win-rate: SFT+DPO vs SFT-only, position-randomized ---
    win_counts = defaultdict(int)  # "dpo" / "sft" / "tie"
    pairwise_examples = []
    for row in rows:
        dpo_is_a = random.random() < 0.5
        text_a = row["dpo"] if dpo_is_a else row["sft"]
        text_b = row["sft"] if dpo_is_a else row["dpo"]

        system_prompt = pairwise_prompts[row["voice"]]
        user_prompt = build_pairwise_user_prompt(row["brief"], row["format"], text_a, text_b)
        try:
            raw = call_with_retry(call_fn, system_prompt, user_prompt, model, 100, args.sleep)
            data = extract_json_object(raw)
            winner_letter = data.get("winner", "").strip().lower()
        except Exception as e:
            print(f"[{row['id']}] pairwise judge failed: {e} -- skipping this row's win-rate contribution", file=sys.stderr)
            continue

        if winner_letter == "tie":
            winner_model = "tie"
        elif winner_letter == "a":
            winner_model = "dpo" if dpo_is_a else "sft"
        elif winner_letter == "b":
            winner_model = "sft" if dpo_is_a else "dpo"
        else:
            print(f"[{row['id']}] unexpected winner {winner_letter!r} -- skipping", file=sys.stderr)
            continue

        win_counts[winner_model] += 1
        pairwise_examples.append({**row, "pairwise_winner": winner_model, "pairwise_reason": data.get("reason", "")})
        time.sleep(args.sleep)

    total_judged = sum(win_counts.values())
    dpo_win_rate = win_counts["dpo"] / total_judged if total_judged else 0.0
    print(f"\nPairwise (SFT+DPO vs SFT-only), n={total_judged}: "
          f"DPO wins {win_counts['dpo']} ({dpo_win_rate:.0%}), "
          f"SFT wins {win_counts['sft']}, ties {win_counts['tie']}")

    # --- 3. rubric scores per variant ---
    rubric_totals = {v: defaultdict(list) for v in ("base", "sft", "dpo")}
    for row in rows:
        for variant in ("base", "sft", "dpo"):
            system_prompt = rubric_prompts[row["voice"]]
            user_prompt = build_rubric_user_prompt(row["brief"], row["format"], row[variant])
            try:
                raw = call_with_retry(call_fn, system_prompt, user_prompt, model, 100, args.sleep)
                scores = parse_rubric_response(raw)
            except Exception as e:
                print(f"[{row['id']}/{variant}] rubric judge failed: {e} -- skipping", file=sys.stderr)
                time.sleep(args.sleep)
                continue
            for dim, val in scores.items():
                rubric_totals[variant][dim].append(val)
            time.sleep(args.sleep)

    rubric_avg = {
        variant: {dim: (sum(vals) / len(vals) if vals else None) for dim, vals in dims.items()}
        for variant, dims in rubric_totals.items()
    }

    # --- 4. write report ---
    examples = random.sample(pairwise_examples, min(args.n_examples, len(pairwise_examples)))

    lines = [
        "# VoiceForge Evaluation Results (Day 10)",
        "",
        f"Evaluated on {len(rows)} held-out test prompts (never seen during SFT or DPO training).",
        "",
        "## Rule-based compliance rate",
        "(banned phrase / CTA rule / length target -- % of completions with zero violations)",
        "",
        "| Variant | Rule-clean rate |",
        "|---|---|",
    ] + [f"| {v.upper()} | {rule_clean_rate[v]:.0%} |" for v in ("base", "sft", "dpo")] + [
        "",
        "## Pairwise preference: SFT+DPO vs. SFT-only",
        f"(LLM-as-judge, position-randomized, n={total_judged})",
        "",
        f"- **SFT+DPO win rate: {dpo_win_rate:.0%}**",
        (f"- SFT-only wins: {win_counts['sft']} ({win_counts['sft']/total_judged:.0%})"
         if total_judged else "- SFT-only wins: n/a"),
        f"- Ties: {win_counts['tie']}",
        "",
        "## Rubric scores (1-5, averaged across test set)",
        "",
        "| Variant | " + " | ".join(RUBRIC_DIMENSIONS) + " |",
        "|---|" + "---|" * len(RUBRIC_DIMENSIONS),
    ]
    for variant in ("base", "sft", "dpo"):
        row_vals = [
            f"{rubric_avg[variant][d]:.2f}" if rubric_avg[variant].get(d) is not None else "n/a"
            for d in RUBRIC_DIMENSIONS
        ]
        lines.append(f"| {variant.upper()} | " + " | ".join(row_vals) + " |")

    lines += ["", "## Qualitative examples", ""]
    for ex in examples:
        lines += [
            f"**[{ex['voice']}] {ex['format']}** -- pairwise winner: `{ex['pairwise_winner']}` ({ex['pairwise_reason']})",
            "",
            f"- BRIEF: {ex['brief']}",
            f"- BASE: {ex['base']}",
            f"- SFT: {ex['sft']}",
            f"- SFT+DPO: {ex['dpo']}",
            "",
        ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
