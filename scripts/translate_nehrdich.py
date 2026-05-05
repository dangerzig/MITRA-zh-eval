#!/usr/bin/env python3
"""
Translate using the exact prompt from Nehrdich et al. (MITRA-zh-eval).
Non-RAG baseline: minimal prompt matching their Appendix A without
dictionary entries or example pairs.

Usage:
    python3 translate_nehrdich.py \
        --input data/chinese_sentences.jsonl \
        --model claude-sonnet-4-20250514 \
        --output translations/sonnet_nehrdich.jsonl \
        --limit 200
"""

import argparse
import json
import os
import sys
import time

import anthropic

# Exact prompt from Appendix A, minus RAG components
SYSTEM_PROMPT = "You are an expert translator of classical Asian languages."
USER_PROMPT = """Now translate the following text to English. Provide only the translation, without any explanation or additional information:

{chinese}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    sentences = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            sentences.append(json.loads(line))
    print(f"Loaded {len(sentences)} sentences")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    done_ids = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])

    remaining = [s for s in sentences if s["id"] not in done_ids]
    print(f"To translate: {len(remaining)} with {args.model}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "a") as out:
        for i, sent in enumerate(remaining):
            try:
                r = client.messages.create(
                    model=args.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user",
                               "content": USER_PROMPT.format(chinese=sent["chinese"])}],
                )
                translation = r.content[0].text.strip()
                result = {
                    "id": sent["id"],
                    "text_id": sent["text_id"],
                    "sentence_idx": sent["sentence_idx"],
                    "chinese": sent["chinese"],
                    "english": translation,
                    "model": args.model,
                    "prompt": "nehrdich_nrag",
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(remaining)}")
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(5)

    print("Done")


if __name__ == "__main__":
    main()
