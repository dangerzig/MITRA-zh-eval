#!/usr/bin/env python3
"""
Extract reference English translations from BDK PDFs and align them
with Chinese source sentences for GEMBA scoring.

Uses a simple approach: extract all English text from the PDF, then
use an LLM to find the best translation match for each Chinese sentence.

Usage:
    python3 extract_references.py
"""

import json
import os
import re
import subprocess
import sys
import time

# Map eval text IDs to BDK PDF paths and page ranges for the translation body
BDK_PDFS = {
    "T01n0001": {
        "pdf": "dBET_T0001_LengthyDiscourses1_2015.pdf",
        "start_page": 20,  # skip front matter
        "end_page": 200,
    },
    "T13n0417": {
        "pdf": "dBET_Pratyutpanna_Surangama_1998_0.pdf",
        "start_page": 15,
        "end_page": 120,
    },
    "T14n0450": {
        "pdf": "dBET_Master-of-Medicine_Candrottara_2018.pdf",
        "start_page": 10,
        "end_page": 80,
    },
    "T17n0842": {
        "pdf": "dBET_ApocryphalScriptures_2005.pdf",
        "start_page": 100,  # Perfect Enlightenment starts later in the volume
        "end_page": 200,
    },
    "T32n1666": {
        "pdf": "dBET_Beta_T1666_AwakeningofFaith_2005.pdf",
        "start_page": 15,
        "end_page": 100,
    },
    "T12n0374": {
        "pdf": "dBET_T0374_NirvanaSutra1_2013_0.pdf",
        "start_page": 20,
        "end_page": 200,
    },
}

BDK_DIR = os.path.expanduser("~/taisho-translation/bdk-pdfs")
DATA_DIR = os.path.expanduser("~/MITRA-zh-eval/data")


def extract_pdf_text(pdf_path, start_page=1, end_page=None):
    """Extract text from a PDF using pdftotext."""
    cmd = ["pdftotext", "-layout"]
    if start_page:
        cmd.extend(["-f", str(start_page)])
    if end_page:
        cmd.extend(["-l", str(end_page)])
    cmd.extend([pdf_path, "-"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error extracting {pdf_path}: {result.stderr}")
        return ""
    return result.stdout


def split_into_sentences(text):
    """Split English text into sentences."""
    # Remove page numbers, headers, footnotes
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


def main():
    # Load Chinese sentences
    chinese_by_text = {}
    with open(os.path.join(DATA_DIR, "chinese_sentences.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            text_id = obj["text_id"]
            chinese_by_text.setdefault(text_id, []).append(obj)

    output_path = os.path.join(DATA_DIR, "reference_translations.jsonl")
    print(f"Extracting reference translations from BDK PDFs...")

    all_refs = []
    for text_id, info in BDK_PDFS.items():
        pdf_path = os.path.join(BDK_DIR, info["pdf"])
        if not os.path.exists(pdf_path):
            print(f"  SKIP {text_id}: PDF not found at {pdf_path}")
            continue

        chinese_sents = chinese_by_text.get(text_id, [])
        if not chinese_sents:
            print(f"  SKIP {text_id}: no Chinese sentences")
            continue

        print(f"  {text_id}: extracting from {info['pdf']}...")
        eng_text = extract_pdf_text(
            pdf_path, info["start_page"], info["end_page"]
        )
        eng_sentences = split_into_sentences(eng_text)
        print(f"    Extracted {len(eng_sentences)} English sentences")

        # Simple sequential alignment: match Chinese sentences to English
        # sentences by position (first N Chinese -> first N English)
        # This is approximate but usable for GEMBA scoring
        n = min(len(chinese_sents), len(eng_sentences))
        for i in range(n):
            ref = {
                "id": chinese_sents[i]["id"],
                "text_id": text_id,
                "chinese": chinese_sents[i]["chinese"],
                "english": eng_sentences[i],
            }
            all_refs.append(ref)

        print(f"    Aligned {n} sentence pairs")

    with open(output_path, "w", encoding="utf-8") as f:
        for ref in all_refs:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_refs)} reference translations")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
