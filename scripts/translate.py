#!/usr/bin/env python3
"""
Translate Chinese sentences using the Anthropic API.

Replicates the MITRA-zh-eval prompt template for sentence-level
Buddhist Chinese to English translation.

Usage:
    python3 translate.py \
        --input ~/MITRA-zh-eval/data/chinese_sentences.jsonl \
        --model claude-sonnet-4-20250514 \
        --output ~/MITRA-zh-eval/translations/sonnet.jsonl

    python3 translate.py \
        --input ~/MITRA-zh-eval/data/chinese_sentences.jsonl \
        --model claude-opus-4-20250514 \
        --output ~/MITRA-zh-eval/translations/opus.jsonl
"""

import argparse
import json
import os
import sys
import time

import anthropic

# Exact prompt from Nehrdich et al. MITRA-zh-eval (Appendix A), non-RAG variant
SYSTEM_PROMPT = "You are an expert translator of classical Asian languages."

USER_PROMPT = """Now translate the following text to English. Provide only the translation, without any explanation or additional information:

{chinese}"""


def translate_batch(client, sentences, model, output_path, batch_size=50):
    """Translate sentences and write results incrementally."""
    # Load any existing results to support resumption
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj["id"])
        print(f"  Resuming: {len(done_ids)} already translated")

    remaining = [s for s in sentences if s["id"] not in done_ids]
    print(f"  To translate: {len(remaining)} sentences with {model}")

    with open(output_path, "a", encoding="utf-8") as out:
        idx = 0
        while idx < len(remaining):
            sent = remaining[idx]
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": USER_PROMPT.format(chinese=sent["chinese"]),
                    }],
                    system=SYSTEM_PROMPT,
                )
                translation = response.content[0].text.strip()

                result = {
                    "id": sent["id"],
                    "text_id": sent["text_id"],
                    "sentence_idx": sent["sentence_idx"],
                    "chinese": sent["chinese"],
                    "english": translation,
                    "model": model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                idx += 1

                if idx % 50 == 0:
                    print(f"    {idx}/{len(remaining)} done")

            except anthropic.RateLimitError:
                print(f"    Rate limited at {idx}, sleeping 30s...")
                time.sleep(30)
                # Do NOT increment idx; retry the same sentence
                continue
            except Exception as e:
                print(f"    Error on {sent['id']}: {e}")
                idx += 1  # Skip this one but log the error
                continue

    total = len(done_ids) + len(remaining)
    print(f"  Complete: {total} translations in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate Buddhist Chinese sentences via Anthropic API."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input JSONL with Chinese sentences.",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="claude-sonnet-4-20250514",
        help="Anthropic model ID.",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output JSONL for translations.",
    )
    args = parser.parse_args()

    # Load sentences
    sentences = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(json.loads(line))
    print(f"Loaded {len(sentences)} sentences")

    # Set up client (requires ANTHROPIC_API_KEY env var)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    translate_batch(client, sentences, args.model, args.output)


if __name__ == "__main__":
    main()
