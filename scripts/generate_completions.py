"""
Day 3: Generate on-voice completions for each brief in data/raw_prompts.jsonl.

The voiceforge/ repo layout:
  scripts/generate_completions.py   <- this file
  docs/voice_guidelines.md          <- read (finalized voice_guidelines_v2.md)
  data/raw_prompts.jsonl            <- read (Day 2 output)
  data/sft_raw.jsonl                <- written (Day 3 output)

Key difference from a flat "dump the whole doc as system prompt" approach:
this script PARSES voice_guidelines.md into its shared sections (voice
routing mechanism, format/length table, CTA rules) and its two "## PERSONA
N: ..." sections, then builds a persona-SPECIFIC system prompt per brief --
using row["voice"] (cozy_crochet / romantic_floral, written by Day 2's
generate_briefs.py) to select the right one. Each API call only sees the
rules that apply to that example, instead of both personas' tone words /
vocabulary / anti-examples diluting every call.

Default provider is Gemini (free tier via Google AI Studio), called through
the OpenAI-compatible Chat Completions endpoint so the rest of the script
doesn't need a separate SDK. OpenAI is kept as an alternate option.

Requires an API key. Put ONE of these in a `.env` file at your repo root
(or export it in your shell):
  GEMINI_API_KEY=AIza...
  OPENAI_API_KEY=sk-...

Install:
  pip install openai python-dotenv

Usage (run from the voiceforge/ repo root):
  python scripts/generate_completions.py \
      --prompts data/raw_prompts.jsonl \
      --voice docs/voice_guidelines.md \
      --out data/sft_raw.jsonl \
      --provider gemini

Note: check https://ai.google.dev/gemini-api/docs/models and
https://ai.google.dev/gemini-api/docs/pricing for the current recommended
model string and free-tier rate limits before running -- the default below
may drift out of date, and this is a fast-moving space.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file at the repo root, if present.
load_dotenv()

# ---------------------------------------------------------------------------
# voice_guidelines.md parsing
# ---------------------------------------------------------------------------

# Maps Day 2's `voice` key -> substrings that identify that persona's
# "## PERSONA N: ..." header in voice_guidelines.md. Update this if you
# rename a persona's header wording.
PERSONA_HEADER_MATCH = {
    "cozy_crochet": ["cozy crochet", "cozy artisan"],
    "romantic_floral": ["romantic floral", "romantic curator"],
}


def split_h2_sections(md_text):
    """Split a markdown doc into (header_line, body_text) pairs on '## ' headers."""
    lines = md_text.splitlines()
    sections = []
    header, body = None, []
    for line in lines:
        if line.startswith("## "):
            if header is not None:
                sections.append((header, "\n".join(body).strip()))
            header, body = line, []
        else:
            body.append(line)
    if header is not None:
        sections.append((header, "\n".join(body).strip()))
    return sections


def parse_voice_doc(md_text):
    """Return (shared_text, {voice_key: persona_text or None})."""
    sections = split_h2_sections(md_text)
    shared_chunks = []
    persona_text = {key: None for key in PERSONA_HEADER_MATCH}

    for header, body in sections:
        header_lower = header.lower()
        matched_key = None
        if "persona" in header_lower:
            for key, needles in PERSONA_HEADER_MATCH.items():
                if any(needle in header_lower for needle in needles):
                    matched_key = key
                    break
        if matched_key:
            persona_text[matched_key] = f"{header}\n{body}"
        else:
            shared_chunks.append(f"{header}\n{body}")

    shared_text = "\n\n".join(shared_chunks)
    return shared_text, persona_text


def build_system_prompts(voice_doc_text):
    """Returns {voice_key: system_prompt_string}, with a safe fallback to
    the full document if a persona section can't be located."""
    shared_text, persona_text = parse_voice_doc(voice_doc_text)

    prompts = {}
    for voice_key in PERSONA_HEADER_MATCH:
        section = persona_text.get(voice_key)
        if section is None:
            print(
                f"WARNING: could not find a '## PERSONA' section matching "
                f"'{voice_key}' -- falling back to the full document for "
                f"this voice. Check your header wording in voice_guidelines.md.",
                file=sys.stderr,
            )
            body = voice_doc_text
        else:
            body = f"{shared_text}\n\n{section}"

        prompts[voice_key] = (
            "You are a professional copywriter working for one specific "
            "small handmade business. Write copy that strictly matches the "
            "brand voice guidelines below. Output ONLY the copy itself, "
            "ready to publish -- no explanations, no labels, no surrounding "
            "quotation marks, and no restating the [Voice: ...] tag.\n\n"
            f"{body}"
        )
    return prompts


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def already_done_ids(out_path):
    if not Path(out_path).exists():
        return set()
    return {
        json.loads(line)["id"]
        for line in Path(out_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


# ---------------------------------------------------------------------------
# Provider calls
#
# Both providers go through the OpenAI python client: for Gemini, it's
# pointed at Google's OpenAI-compatible endpoint via base_url. This keeps
# one dependency (openai) instead of two separate SDKs.
# ---------------------------------------------------------------------------

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
    client = OpenAI()  # reads OPENAI_API_KEY from env
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
    "gemini": {
        "call_fn": call_gemini,
        "default_model": "gemini-3.5-flash-lite",
        "key_env_var": "GEMINI_API_KEY",
    },
    "openai": {
        "call_fn": call_openai,
        "default_model": "gpt-4o-mini",
        "key_env_var": "OPENAI_API_KEY",
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/raw_prompts.jsonl")
    ap.add_argument("--voice", default="docs/voice_guidelines.md")
    ap.add_argument("--out", default="data/sft_raw.jsonl")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ap.add_argument("--model", default=None, help="overrides the provider default")
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument(
        "--sleep", type=float, default=2.0,
        help="seconds between calls -- free tiers are typically low "
             "requests-per-minute, so keep this conservative and raise it "
             "if you hit 429s",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap number of briefs (cheap test run)")
    args = ap.parse_args()

    provider = PROVIDERS[args.provider]
    if provider["key_env_var"] not in os.environ:
        sys.exit(
            f"Set {provider['key_env_var']} before running -- either export it "
            f"or add it to a .env file at the repo root (see docstring)."
        )
    call_fn = provider["call_fn"]
    model = args.model or provider["default_model"]

    briefs = load_jsonl(args.prompts)
    if args.limit:
        briefs = briefs[: args.limit]

    voice_doc_text = Path(args.voice).read_text(encoding="utf-8")
    system_prompts = build_system_prompts(voice_doc_text)
    for key, prompt in system_prompts.items():
        print(f"[{key}] system prompt: {len(prompt)} chars")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = already_done_ids(args.out)
    print(f"{len(done)} of {len(briefs)} already completed. Resuming...")

    with open(args.out, "a", encoding="utf-8") as f:
        for i, row in enumerate(briefs):
            if row["id"] in done:
                continue

            voice_key = row.get("voice")
            system_prompt = system_prompts.get(voice_key)
            if system_prompt is None:
                print(f"[{row['id']}] ERROR: unknown voice '{voice_key}' -- skipping", file=sys.stderr)
                continue

            user_prompt = row["brief"]
            try:
                completion = call_fn(system_prompt, user_prompt, model, args.max_tokens)
            except Exception as e:
                print(f"[{row['id']}] ERROR: {e} -- skipping, rerun script to retry", file=sys.stderr)
                continue

            out_row = {
                "id": row["id"],
                "voice": row["voice"],
                "voice_tag": row["voice_tag"],
                "format": row["format"],
                "product": row["product"],
                "angle": row["angle"],
                "brief": row["brief"],
                "completion": completion,
            }
            f.write(json.dumps(out_row) + "\n")
            f.flush()

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(briefs)} done")
            time.sleep(args.sleep)

    print(f"Done. Output written to {args.out}")


if __name__ == "__main__":
    main()
