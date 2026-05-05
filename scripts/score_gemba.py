#!/usr/bin/env python3
"""
Score translations using GEMBA (LLM-based evaluation).

Uses Gemini 2.5 Flash as the judge model, following the MITRA-zh-eval
methodology. Supports both reference-based GEMBA and reference-free
GEMBA* scoring.

Usage:
    # Reference-free mode (GEMBA*)
    python3 score_gemba.py \
        --translations translations/sonnet.jsonl \
        --reference-free \
        --output results/sonnet_scores.jsonl
"""

import argparse
import json
import os
import re
import sys
import time

from google import genai
from google.genai import types

# Exact GEMBA-DA_ref prompt from Kocmi & Federmann 2023
# (https://github.com/MicrosoftTranslator/GEMBA)
GEMBA_PROMPT = """Score the following translation from Chinese to English with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".

Chinese source: "{source}"
English human reference: "{reference}"
English machine translation: "{translation}"
Score: """

# Exact GEMBA-DA prompt (reference-free variant)
GEMBA_FREE_PROMPT = """Score the following translation from Chinese to English on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".

Chinese source: "{source}"
English translation: "{translation}"
Score: """


def score_translation(client, source, translation, reference=None):
    """Score a single translation using GEMBA."""
    if reference:
        prompt = GEMBA_PROMPT.format(
            source=source, reference=reference, translation=translation,
        )
    else:
        prompt = GEMBA_FREE_PROMPT.format(
            source=source, translation=translation,
        )

    try:
        response = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
        )
        text = response.text.strip()
        # Strict parsing: expect just a number, optionally followed by /100
        # First try exact match (just a number)
        exact = re.match(r"^(\d{1,3})(?:/100)?\.?$", text)
        if exact:
            score = int(exact.group(1))
            return min(max(score, 0), 100)
        # Fallback: find "Score: N" or just a lone number
        scored = re.search(r"(?:score[:\s]*)?(\d{1,3})(?:/100)?", text, re.IGNORECASE)
        if scored:
            score = int(scored.group(1))
            if 0 <= score <= 100:
                return score
        print(f"    Could not parse score from: {text[:60]}")
        return None
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            time.sleep(5)
            return None  # Will be retried on next run via resumption
        print(f"    Gemini error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Score translations with GEMBA using Gemini 2.5 Flash."
    )
    parser.add_argument("--translations", "-t", type=str, required=True)
    parser.add_argument("--references", "-r", type=str, default=None)
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--gemini-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GEMINI_API_KEY or pass --gemini-key", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # Load translations
    translations = []
    with open(args.translations, "r", encoding="utf-8") as f:
        for line in f:
            translations.append(json.loads(line))
    print(f"Loaded {len(translations)} translations")

    # Load references if provided
    refs = {}
    if args.references and not args.reference_free:
        with open(args.references, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                refs[obj["id"]] = obj["english"]
        print(f"Loaded {len(refs)} reference translations")

    # Load existing scores for resumption
    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj["id"])
        print(f"Resuming: {len(done_ids)} already scored")

    remaining = [t for t in translations if t["id"] not in done_ids]

    # For reference-based mode, filter to only sentences with references
    if not args.reference_free and refs:
        has_ref = [t for t in remaining if t["id"] in refs]
        skipped = len(remaining) - len(has_ref)
        remaining = has_ref
        print(f"Reference-based mode: {len(remaining)} sentences with refs, {skipped} skipped (no ref)")
    print(f"To score: {len(remaining)}")

    scores = []
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out:
        for i, t in enumerate(remaining):
            ref = refs.get(t["id"]) if not args.reference_free else None

            score = score_translation(
                client,
                source=t["chinese"],
                translation=t["english"],
                reference=ref if not args.reference_free else None,
            )

            if score is not None:
                result = {
                    "id": t["id"],
                    "text_id": t["text_id"],
                    "model": t.get("model", "unknown"),
                    "score": score,
                    "mode": "reference-free" if args.reference_free else "reference",
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                scores.append(score)

            if (i + 1) % 100 == 0:
                avg = sum(scores[-100:]) / len(scores[-100:]) if scores else 0
                print(f"    {i + 1}/{len(remaining)} done, recent avg: {avg:.1f}")

            time.sleep(0.15)  # Rate limiting

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nAverage GEMBA score: {avg:.2f} (n={len(scores)})")
    print(f"Results in: {args.output}")


if __name__ == "__main__":
    main()
