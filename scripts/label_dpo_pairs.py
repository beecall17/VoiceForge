"""
Day 8: Label DPO candidate pairs as chosen/rejected.

Two-tier labeling, in order of preference:
  1. RULE-BASED (reuses Day 4's exact banned-phrase/CTA/length checks from
     clean_and_split.py): if one candidate has strictly fewer objective
     rule violations than the other, it wins. This is more reliable than
     an LLM judge for the objective stuff -- a rule either fires or it
     doesn't -- and it's free.
  2. LLM JUDGE (only for pairs where rule violations tie, usually both
     candidates at zero issues): asks a judge model to compare A vs B
     against the SAME persona-specific rubric text used for teacher
     generation in Day 3, scored on voice adherence specifically --
     not generic "which reads better," since a cleaner-sounding line
     that skips a required CTA or drops a concrete detail is still a
     rejection under this project's own rules.

Pairs the judge calls a genuine tie are written to a separate discarded
file rather than forced into a label -- a weak/ambiguous pair teaches
DPO nothing and just adds noise.

Output: data/dpo_pairs.jsonl        (id, voice, brief, chosen, rejected, ...)
        data/dpo_pairs_discarded.jsonl  (ties, for your own reference)

Requires an API key for the judge fallback. Set ONE of:
  GEMINI_API_KEY=...   (in .env, same as Day 3)
  OPENAI_API_KEY=...

Usage (run from the voiceforge/ repo root):
  python scripts/label_dpo_pairs.py \\
      --candidates data/dpo_candidates.jsonl \\
      --voice docs/voice_guidelines.md \\
      --out data/dpo_pairs.jsonl \\
      --provider gemini
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# reuse Day 4's rule constants and helpers -- one source of truth for what
# counts as a violation, instead of a second, possibly-drifted copy here.
sys.path.insert(0, str(Path(__file__).parent))
from clean_and_split import (  # noqa: E402
    CLICHE_BLOCKLIST, FORMAT_WORD_BOUNDS, CTA_KEYWORDS,
    CTA_REQUIRED_FORMATS, CTA_FORBIDDEN_FORMATS, contains_any, word_count,
)
from generate_completions import parse_voice_doc  # noqa: E402


# ---------------------------------------------------------------------------
# Tier 1: rule-based check (same rules as Day 4, applied to a raw candidate
# string rather than an already-cleaned row)
# ---------------------------------------------------------------------------

def check_candidate(text, voice, fmt):
    issues = []
    wc = word_count(text)
    lo, hi = FORMAT_WORD_BOUNDS.get(fmt, (1, 10_000))
    if wc < lo or wc > hi:
        issues.append(f"length_out_of_range(words={wc})")

    hits = contains_any(text.lower(), CLICHE_BLOCKLIST.get(voice, []))
    if hits:
        issues.append(f"banned_phrase({','.join(hits)})")

    has_cta = bool(contains_any(text.lower(), CTA_KEYWORDS))
    if fmt in CTA_REQUIRED_FORMATS and not has_cta:
        issues.append("missing_required_cta")
    if fmt in CTA_FORBIDDEN_FORMATS and has_cta:
        issues.append("cta_present_but_forbidden")

    return issues


# ---------------------------------------------------------------------------
# Tier 2: LLM judge (only called when rule-based issue counts tie)
# ---------------------------------------------------------------------------

def build_judge_system_prompts(voice_doc_text):
    shared_text, persona_text = parse_voice_doc(voice_doc_text)
    prompts = {}
    for voice_key, section in persona_text.items():
        body = f"{shared_text}\n\n{section}" if section else voice_doc_text
        prompts[voice_key] = (
            "You are an expert brand-voice reviewer. You will be shown two candidate "
            "marketing copy completions (A and B) written for the same brief, in the "
            "brand voice described below. Judge ONLY on adherence to this voice guide "
            "-- tone words, vocabulary, banned phrases, emoji policy, CTA rule, and any "
            "required concrete detail. Do not reward generic 'good writing' or "
            "fluency if it comes at the expense of the voice guide's specific rules.\n\n"
            'Respond with ONLY a JSON object, no other text: '
            '{"winner": "A", "reason": "<one sentence>"} or '
            '{"winner": "B", "reason": "<one sentence>"} or '
            '{"winner": "tie", "reason": "<one sentence>"}. '
            'Use "tie" only if both are genuinely equally on-voice or equally flawed '
            '-- do not default to tie just because both are decent.\n\n'
            f"{body}"
        )
    return prompts


def build_judge_user_prompt(brief, candidate_a, candidate_b):
    return (
        f"BRIEF: {brief}\n\n"
        f"CANDIDATE A: {candidate_a}\n\n"
        f"CANDIDATE B: {candidate_b}\n\n"
        "Which candidate better fits the brand voice guide above? JSON only."
    )


def extract_json_object(raw_text):
    """Robustly pull a {...} object out of a model response, tolerant of
    code fences and leading/trailing commentary around the JSON -- reused
    by both pairwise-judge parsing here and rubric-score parsing in
    evaluate_and_report.py, so the fix lives in exactly one place."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in response: {raw_text!r}")
    return json.loads(cleaned[start:end + 1])


def parse_judge_response(raw_text):
    data = extract_json_object(raw_text)
    winner = data.get("winner", "").strip().lower()
    if winner not in ("a", "b", "tie"):
        raise ValueError(f"unexpected winner value: {winner!r}")
    return winner, data.get("reason", "")


def call_gemini(system_prompt, user_prompt, model, max_tokens):
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def call_openai(system_prompt, user_prompt, model, max_tokens):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


PROVIDERS = {
    "gemini": {"call_fn": call_gemini, "default_model": "gemini-3.5-flash-lite", "key_env_var": "GEMINI_API_KEY"},
    "openai": {"call_fn": call_openai, "default_model": "gpt-4o-mini", "key_env_var": "OPENAI_API_KEY"},
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def already_done_ids(*paths):
    ids = set()
    for path in paths:
        if Path(path).exists():
            ids |= {json.loads(l)["id"] for l in Path(path).read_text().splitlines() if l.strip()}
    return ids


def make_pair_row(row, winner, chosen_source, rejected_source, method, reason):
    candidate = {"a": row["candidate_a"], "b": row["candidate_b"]}
    return {
        "id": row["id"],
        "voice": row["voice"],
        "voice_tag": row["voice_tag"],
        "format": row["format"],
        "product": row["product"],
        "angle": row["angle"],
        "brief": row["brief"],
        "chosen": candidate[chosen_source],
        "rejected": candidate[rejected_source],
        "chosen_source": chosen_source,
        "label_method": method,
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/dpo_candidates.jsonl")
    ap.add_argument("--voice", default="docs/voice_guidelines.md")
    ap.add_argument("--out", default="data/dpo_pairs.jsonl")
    ap.add_argument("--discarded_out", default="data/dpo_pairs_discarded.jsonl")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--near_duplicate_threshold", type=float, default=0.9,
                     help="candidate pairs with text similarity at or above this are discarded as low-signal")
    args = ap.parse_args()

    provider = PROVIDERS[args.provider]
    if provider["key_env_var"] not in os.environ:
        sys.exit(f"Set {provider['key_env_var']} before running (see docstring).")
    call_fn = provider["call_fn"]
    model = args.model or provider["default_model"]

    rows = load_jsonl(args.candidates)
    voice_doc_text = Path(args.voice).read_text(encoding="utf-8")
    judge_prompts = build_judge_system_prompts(voice_doc_text)

    done = already_done_ids(args.out, args.discarded_out)
    print(f"{len(done)} of {len(rows)} already labeled. Resuming...")

    auto_count, judge_count, discard_count, judge_fail_count = 0, 0, 0, 0

    out_f = open(args.out, "a", encoding="utf-8")
    discard_f = open(args.discarded_out, "a", encoding="utf-8")

    try:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue

            issues_a = check_candidate(row["candidate_a"], row["voice"], row["format"])
            issues_b = check_candidate(row["candidate_b"], row["voice"], row["format"])

            similarity = difflib.SequenceMatcher(
                None, row["candidate_a"].lower(), row["candidate_b"].lower()
            ).ratio()
            if similarity >= args.near_duplicate_threshold:
                # near-identical text (same idea as Day 7's diversity check)
                # carries almost no preference signal regardless of which
                # rule/judge outcome would otherwise apply -- discard rather
                # than train on a near-no-op pair.
                discard_f.write(json.dumps({
                    **row, "discard_reason": f"near_identical(similarity={similarity:.2f})",
                }) + "\n")
                discard_f.flush()
                discard_count += 1
                continue

            if len(issues_a) != len(issues_b) and min(len(issues_a), len(issues_b)) == 0:
                # a clean auto-rule win: the winner has ZERO issues, not just
                # fewer than the loser. A "fewer issues" winner that still
                # violates a rule is not a valid chosen target -- it would
                # train the model to reproduce that violation, just a
                # slightly less bad version of it. Discard those instead.
                winner = "a" if len(issues_a) < len(issues_b) else "b"
                loser = "b" if winner == "a" else "a"
                reason = f"rule-based: A issues={issues_a or 'none'}, B issues={issues_b or 'none'}"
                out_row = make_pair_row(row, winner, winner, loser, "auto_rule", reason)
                out_f.write(json.dumps(out_row) + "\n")
                out_f.flush()
                auto_count += 1
                continue

            if len(issues_a) > 0 and len(issues_b) > 0:
                # both flawed (whether equally or not) -- neither is a valid
                # chosen target. Flag for regeneration/manual review rather
                # than picking "the less bad one" as a training target.
                discard_f.write(json.dumps({
                    **row, "discard_reason": f"both_flawed: A={issues_a}, B={issues_b}",
                }) + "\n")
                discard_f.flush()
                discard_count += 1
                continue

            # tie on rule violations (both zero issues) -- ask the judge
            system_prompt = judge_prompts.get(row["voice"])
            if system_prompt is None:
                print(f"[{row['id']}] ERROR: unknown voice '{row['voice']}' -- skipping", file=sys.stderr)
                continue

            user_prompt = build_judge_user_prompt(row["brief"], row["candidate_a"], row["candidate_b"])
            try:
                raw = call_fn(system_prompt, user_prompt, model, args.max_tokens)
                winner, reason = parse_judge_response(raw)
            except Exception as e:
                # Transient (quota/rate-limit) or one-off parse failures
                # shouldn't be treated the same as a genuine "tie" verdict --
                # that would permanently throw away a pair that never
                # actually got judged. Skip without writing to EITHER output
                # file, so this id is NOT marked done and a rerun retries it.
                print(f"[{row['id']}] judge call/parse failed: {e} -- skipping for retry "
                      f"(rerun this script later to pick it back up)", file=sys.stderr)
                judge_fail_count += 1
                time.sleep(args.sleep)
                continue

            if winner == "tie":
                discard_f.write(json.dumps({**row, "discard_reason": f"judge_tie: {reason}"}) + "\n")
                discard_f.flush()
                discard_count += 1
            else:
                loser = "b" if winner == "a" else "a"
                out_row = make_pair_row(row, winner, winner, loser, "llm_judge", reason)
                out_f.write(json.dumps(out_row) + "\n")
                out_f.flush()
                judge_count += 1

            time.sleep(args.sleep)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} processed")
    finally:
        out_f.close()
        discard_f.close()

    print(f"\nDone.")
    print(f"  Auto-labeled by rules: {auto_count}")
    print(f"  Labeled by LLM judge:  {judge_count}")
    print(f"  Discarded (near-identical or both-flawed or tie): {discard_count}")
    print(f"  Skipped for retry (judge call/parse failed): {judge_fail_count}")
    if judge_fail_count:
        print(f"  -> rerun this exact command to retry those {judge_fail_count} rows "
              f"(not marked done, so they'll be picked back up automatically). "
              f"If it's a quota error, consider raising --sleep first.")
    print(f"Output: {args.out}")
    print(f"Discarded: {args.discarded_out}")


if __name__ == "__main__":
    main()