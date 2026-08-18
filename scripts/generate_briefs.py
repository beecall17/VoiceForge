"""
Day 2 (v2): Generate diverse, voice-tagged campaign briefs for VoiceForge.

Matches voice_guidelines.md:
  - Every brief is prefixed with its persona's [Voice: ...] tag, since the
    tag is trained IN the input (single adapter learns to condition on it),
    not used as a UI-side system-prompt switch.
  - Formats are the artisan-real channels from the guidelines' length table
    (IG caption, TikTok hook, TikTok caption, Etsy listing, promo/restock
    announcements), not generic e-commerce formats.
  - Product categories are pulled directly from each persona's expanded
    list in voice_guidelines.md.
  - Output is balanced 50/50 between the two voices by construction, not
    left to chance shuffling -- an imbalanced SFT set would bias the model
    toward whichever voice happened to be overrepresented.

Produces `raw_prompts.jsonl`, one JSON object per line:
  {"id", "voice", "voice_tag", "format", "product", "angle",
   "length_hint", "cta_instruction", "brief"}

No API key required.

Usage:
  python scripts/generate_briefs.py --n 250 --out data/raw_prompts.jsonl
"""

import json
import random
import itertools
import argparse
from pathlib import Path
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# Personas (from voice_guidelines.md)
# ---------------------------------------------------------------------------
PERSONAS = {
    "cozy_crochet": {
        "tag": "[Voice: Cozy Crochet]",
        "products": [
            "crochet keychain", "crochet scarf", "crochet flower pot",
            "amigurumi plushie", "chunky lapghan blanket", "crochet coaster set",
            "crochet bag charm", "crochet cardigan", "custom reference order",
        ],
    },
    "romantic_floral": {
        "tag": "[Voice: Romantic Floral]",
        "products": [
            "ribbon bouquet", "pipe-cleaner bouquet", "money bouquet",
            "handwritten-note gift pack", "anniversary bouquet special",
            "graduation bouquet special", "Valentine's Day bouquet",
            "Mother's Day bouquet", "sympathy arrangement",
            "wedding/proposal bouquet", "teacher appreciation gift",
            "custom reference order",
        ],
    },
}

# ---------------------------------------------------------------------------
# Formats + length targets (shared table from voice_guidelines.md)
# cta: True = always include, False = never include, "optional" = light touch
# ---------------------------------------------------------------------------
FORMATS = {
    "ig_caption": {
        "label": "an Instagram caption",
        "length_hint": "20-45 words, 2-4 sentences",
        "cta": True,
        "extra": "",
    },
    "tiktok_hook": {
        "label": "a TikTok video hook (opening line only)",
        "length_hint": "under 10 words, 1 line",
        "cta": False,
        "extra": "This is only the first line viewers see before the video plays -- pure scroll-stopper, not a full post.",
    },
    "tiktok_caption": {
        "label": "a TikTok caption",
        "length_hint": "15-30 words",
        "cta": "optional",
        "extra": "",
    },
    "etsy_listing": {
        "label": "an Etsy-style product listing description",
        "length_hint": "40-70 words, 3-5 sentences",
        "cta": True,
        "extra": "Must include at least one concrete, practical detail (turnaround time, size, material, or care note).",
    },
    "promo_announcement": {
        "label": "a story/promo announcement",
        "length_hint": "10-20 words",
        "cta": "optional",
        "extra": "Time-sensitive framing, but not salesy or hypey.",
    },
    "restock_announcement": {
        "label": "a restock/drop announcement",
        "length_hint": "15-30 words",
        "cta": True,
        "extra": "Frame it as 'back' or 'new'.",
    },
}

CTA_INSTRUCTION = {
    True: "Include a natural CTA encouraging DMs for orders or custom requests.",
    False: "Do NOT include any CTA -- this format has none.",
    "optional": "A light CTA is okay here if it fits naturally, but isn't required.",
}

# ---------------------------------------------------------------------------
# Campaign angles (shared -- the "why now" framing, independent of product)
# ---------------------------------------------------------------------------
ANGLES = [
    "new design reveal",
    "restock of a bestseller",
    "custom order / reference-photo feature",
    "behind-the-scenes of how it's made",
    "gift guide for an upcoming occasion",
    "customer feature / happy customer story",
    "care tip or fun fact about the piece",
    "limited custom-order slots reminder",
]


def article_for(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def make_brief(voice_tag, fmt, product, angle):
    art = article_for(product)
    parts = [
        voice_tag,
        f"Write {fmt['label']} ({fmt['length_hint']}) for {art} {product}.",
        f"Campaign angle: {angle}.",
        CTA_INSTRUCTION[fmt["cta"]],
    ]
    if fmt["extra"]:
        parts.append(fmt["extra"])
    return " ".join(parts)


def generate_for_persona(persona_key, n):
    persona = PERSONAS[persona_key]
    combos = list(itertools.product(persona["products"], FORMATS.items(), ANGLES))
    random.shuffle(combos)

    rows = []
    for product, (fmt_key, fmt), angle in combos:
        rows.append({
            "id": f"{persona_key}_{len(rows):04d}",
            "voice": persona_key,
            "voice_tag": persona["tag"],
            "format": fmt_key,
            "product": product,
            "angle": angle,
            "length_hint": fmt["length_hint"],
            "cta_instruction": CTA_INSTRUCTION[fmt["cta"]],
            "brief": make_brief(persona["tag"], fmt, product, angle),
        })
        if len(rows) >= n:
            break
    return rows


def generate(n_total, out_path):
    per_persona = n_total // len(PERSONAS)
    rows = []
    for persona_key in PERSONAS:
        rows.extend(generate_for_persona(persona_key, per_persona))

    random.shuffle(rows)  # interleave voices so training order isn't blocked

    Path(out_path).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} briefs to {out_path}")
    print("Voice balance:", Counter(r["voice"] for r in rows))
    print("Format coverage:", Counter(r["format"] for r in rows))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="total briefs, split evenly across voices")
    ap.add_argument("--out", default="raw_prompts.jsonl")
    args = ap.parse_args()
    generate(args.n, args.out)
