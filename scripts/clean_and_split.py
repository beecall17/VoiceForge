"""
Day 4: Clean, quality-check, and split data/sft_raw.jsonl.

Pipeline (matches the Day 4 schedule -- automated checks first, then a
human spot-check pass, then split):

  1. Auto-clean: strip stray wrapping quotes/whitespace.
  2. Auto-flag hard failures: banned clichés, missing/extra CTAs relative
     to each format's rule, way-off-target length, empty/too-short
     completions, exact-duplicate completions. Flagged rows are removed
     from the working set and written to data/sft_flagged.jsonl with
     reasons attached, for your own reference (not for retraining).
  3. Write a random ~20% sample of what's LEFT to
     data/sft_manual_review_sample.jsonl -- automated checks catch
     objective problems, not "this is technically fine but sounds
     generic," so read this sample yourself. If you find bad ones, put
     their ids (one per line) in a text file and rerun with
     --exclude_ids_file to drop them before splitting.
  4. Stratified 80/10/10 split by voice into data/train.jsonl,
     data/val.jsonl, data/test.jsonl, plus a data/curation_report.md
     summary you can drop straight into your README.

Run from the voiceforge/ repo root:
  python scripts/clean_and_split.py
  # ... read data/sft_manual_review_sample.jsonl, write bad ids to a file ...
  python scripts/clean_and_split.py --exclude_ids_file data/manual_rejects.txt
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Rules mirrored from voice_guidelines.md -- keep these in sync if the
# guidelines change. (Kept as plain Python here rather than markdown-parsed
# since these are short, stable lists and explicit is safer than clever.)
# ---------------------------------------------------------------------------

CLICHE_BLOCKLIST = {
    "cozy_crochet": [
        "elevate your accessory game", "must-have staple for your wardrobe",
        "unleash your style", "the ultimate fashion statement",
        "luxurious", "premium quality", "synergy", "game-changer",
        "game changer", "exclusive drop",
    ],
    "romantic_floral": [
        "unlock the secrets of romance", "unrivaled elegance for your lifestyle",
        "revolutionary floral technology", "cheap", "standard", "bulk",
        "commercial floral", "flash sale",
    ],
}

# (min_words, max_words) -- deliberately a bit wider than the brief's
# stated hint, since a strong completion that's a few words over shouldn't
# be auto-rejected. Grossly-off completions still get caught.
FORMAT_WORD_BOUNDS = {
    "ig_caption": (12, 55),
    "tiktok_hook": (2, 16),
    "tiktok_caption": (8, 40),
    "etsy_listing": (25, 90),
    "promo_announcement": (4, 28),
    "restock_announcement": (8, 40),
}

CTA_KEYWORDS = [
    "dm", "message us", "message me", "send us", "send a", "order now",
    "order today", "shop now", "shop the", "link in bio", "comment below",
    "swipe up", "tap", "visit", "book your", "reach out",
]

# formats that must always carry a CTA per voice_guidelines.md's shared
# CTA rules table; tiktok_hook must NEVER carry one; the rest are optional
CTA_REQUIRED_FORMATS = {"ig_caption", "etsy_listing", "restock_announcement"}
CTA_FORBIDDEN_FORMATS = {"tiktok_hook"}


def strip_wrapping_quotes(text):
    text = text.strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'" and text[0] == text[-1]:
        return text[1:-1].strip()
    return text


def word_count(text):
    return len(text.split())


def contains_any(text_lower, phrases):
    """Word/phrase-boundary match -- plain substring matching would flag
    e.g. 'dm' as present inside 'handmade' (han-DM-ade), which silently
    masks real CTA violations. \\b works correctly for multi-word phrases
    too since spaces aren't word characters."""
    hits = []
    for p in phrases:
        if re.search(r"\b" + re.escape(p) + r"\b", text_lower):
            hits.append(p)
    return hits


def check_row(row, seen_completions):
    """Returns a list of failure reasons (empty list = passed)."""
    reasons = []
    completion = row["completion"]
    fmt = row["format"]
    voice = row["voice"]

    if not completion or len(completion.strip()) < 8:
        reasons.append("empty_or_too_short")
        return reasons  # no point checking further

    wc = word_count(completion)
    lo, hi = FORMAT_WORD_BOUNDS.get(fmt, (1, 10_000))
    if wc < lo or wc > hi:
        reasons.append(f"length_out_of_range(words={wc}, expected {lo}-{hi})")

    hits = contains_any(completion.lower(), CLICHE_BLOCKLIST.get(voice, []))
    if hits:
        reasons.append(f"banned_phrase({', '.join(hits)})")

    has_cta = bool(contains_any(completion.lower(), CTA_KEYWORDS))
    if fmt in CTA_REQUIRED_FORMATS and not has_cta:
        reasons.append("missing_required_cta")
    if fmt in CTA_FORBIDDEN_FORMATS and has_cta:
        reasons.append("cta_present_but_forbidden")

    norm = re.sub(r"[^a-z0-9 ]", "", completion.lower()).strip()
    if norm in seen_completions:
        reasons.append(f"exact_duplicate_of({seen_completions[norm]})")
    else:
        seen_completions[norm] = row["id"]

    return reasons


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def stratified_split(rows, train_frac=0.8, val_frac=0.1, seed=42):
    """Split independently within each `voice` group so both personas stay
    balanced across train/val/test, not just in the pooled dataset."""
    rng = random.Random(seed)
    by_voice = defaultdict(list)
    for r in rows:
        by_voice[r["voice"]].append(r)

    train, val, test = [], [], []
    for voice, group in by_voice.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/sft_raw.jsonl")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--review_sample_frac", type=float, default=0.20)
    ap.add_argument("--exclude_ids_file", default=None,
                     help="text file, one id per line, of manually-rejected rows to drop before splitting")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = load_jsonl(args.raw)
    print(f"Loaded {len(rows)} rows from {args.raw}")

    # --- auto-clean (non-destructive fixes) ---
    for r in rows:
        r["completion"] = strip_wrapping_quotes(r["completion"].strip())

    # --- auto-flag hard failures ---
    seen_completions = {}
    passed, flagged = [], []
    for r in rows:
        reasons = check_row(r, seen_completions)
        if reasons:
            flagged.append({**r, "flag_reasons": reasons})
        else:
            passed.append(r)

    write_jsonl(flagged, out_dir / "sft_flagged.jsonl")
    print(f"Auto-flagged {len(flagged)} rows -> {out_dir / 'sft_flagged.jsonl'}")

    reason_counts = Counter()
    for r in flagged:
        for reason in r["flag_reasons"]:
            reason_counts[reason.split("(")[0]] += 1
    print("Flag reason breakdown:", dict(reason_counts))

    # --- apply manual exclusions from a prior review pass, if provided ---
    excluded_ids = set()
    if args.exclude_ids_file and Path(args.exclude_ids_file).exists():
        excluded_ids = {
            line.strip() for line in Path(args.exclude_ids_file).read_text().splitlines() if line.strip()
        }
        before = len(passed)
        passed = [r for r in passed if r["id"] not in excluded_ids]
        print(f"Excluded {before - len(passed)} rows via {args.exclude_ids_file}")

    write_jsonl(passed, out_dir / "sft_clean.jsonl")
    print(f"Clean set: {len(passed)} rows -> {out_dir / 'sft_clean.jsonl'}")

    # --- human review sample (only meaningful before exclusions are applied) ---
    if not args.exclude_ids_file:
        sample_n = max(1, int(len(passed) * args.review_sample_frac))
        sample = random.sample(passed, min(sample_n, len(passed)))
        write_jsonl(sample, out_dir / "sft_manual_review_sample.jsonl")
        print(f"Wrote {len(sample)} rows ({args.review_sample_frac:.0%}) for manual review -> "
              f"{out_dir / 'sft_manual_review_sample.jsonl'}")
        print(
            "\nNext step: read that file, note the ids of anything off-voice or generic, "
            "put one id per line in a .txt file, then rerun with --exclude_ids_file to finalize."
        )
        return  # don't split yet -- wait for the human pass

    # --- split (only runs once you've done the exclude_ids_file pass) ---
    train, val, test = stratified_split(passed, args.train_frac, args.val_frac)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    voice_counts = lambda rs: dict(Counter(r["voice"] for r in rs))
    format_counts = lambda rs: dict(Counter(r["format"] for r in rs))

    report = f"""# Day 4 Curation Report

- Raw completions: {len(rows)}
- Auto-flagged (removed): {len(flagged)} — {dict(reason_counts)}
- Manually excluded: {len(excluded_ids)}
- Final clean set: {len(passed)}

## Split sizes
| Split | Rows | Voice balance |
|---|---|---|
| train | {len(train)} | {voice_counts(train)} |
| val | {len(val)} | {voice_counts(val)} |
| test | {len(test)} | {voice_counts(test)} |

## Format coverage (train)
{format_counts(train)}
"""
    (out_dir / "curation_report.md").write_text(report, encoding="utf-8")
    print(f"\nSplit complete: train={len(train)} val={len(val)} test={len(test)}")
    print(f"Report written to {out_dir / 'curation_report.md'}")


if __name__ == "__main__":
    main()
