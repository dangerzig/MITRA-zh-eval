#!/usr/bin/env python3
"""
Align BDK reference translations with Chinese source sentences using
embedding-based similarity (sentence-transformers).

Uses cosine similarity between multilingual sentence embeddings to find
the best English translation match for each Chinese sentence, replacing
the broken Bertalign wrapper.

Usage:
    python3 align_references.py
"""

import json
import os
import re
import subprocess

import numpy as np
from sentence_transformers import SentenceTransformer

# Map eval text IDs to BDK PDF paths and page ranges
BDK_PDFS = {
    "T01n0001": {
        "pdf": "dBET_T0001_LengthyDiscourses1_2015.pdf",
        "start_page": 20,
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
        "start_page": 100,
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

# Minimum cosine similarity to accept an alignment
MIN_SIMILARITY = 0.4


def extract_pdf_text(pdf_path, start_page=1, end_page=None):
    """Extract text from a PDF using pdftotext."""
    cmd = ["pdftotext", "-layout"]
    if start_page:
        cmd.extend(["-f", str(start_page)])
    if end_page:
        cmd.extend(["-l", str(end_page)])
    cmd.extend([pdf_path, "-"])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def clean_and_split_english(text):
    """Clean PDF text and split into sentences."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\d{1,4}$", line):
            continue
        if len(line) < 10:
            continue
        if re.match(r"^\d+\.\s", line):
            continue
        cleaned.append(line)
    full = " ".join(cleaned)
    # Split on sentence boundaries
    sents = re.split(r'(?<=[.!?])\s+', full)
    return [s.strip() for s in sents if len(s.strip()) > 15]


def align_by_embedding(model, zh_sents, en_sents):
    """Align Chinese to English sentences using cosine similarity.

    For each Chinese sentence, find the best-matching English sentence
    using multilingual sentence embeddings. Enforces monotonicity:
    matches must be in order (no crossing alignments).
    """
    zh_emb = model.encode(zh_sents, normalize_embeddings=True)
    en_emb = model.encode(en_sents, normalize_embeddings=True)

    # Cosine similarity matrix
    sim = np.dot(zh_emb, en_emb.T)

    # Greedy monotonic alignment
    alignments = []
    en_cursor = 0
    for i in range(len(zh_sents)):
        # Search forward from cursor in a window
        window = min(len(en_sents), en_cursor + 50)
        if en_cursor >= len(en_sents):
            break
        best_j = en_cursor + np.argmax(sim[i, en_cursor:window])
        best_score = sim[i, best_j]
        if best_score >= MIN_SIMILARITY:
            alignments.append((i, best_j, float(best_score)))
            en_cursor = best_j + 1  # Move past this match

    return alignments


def main():
    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Load Chinese sentences grouped by text
    chinese_by_text = {}
    with open(os.path.join(DATA_DIR, "chinese_sentences.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            text_id = obj["text_id"]
            chinese_by_text.setdefault(text_id, []).append(obj)

    output_path = os.path.join(DATA_DIR, "reference_translations.jsonl")
    print("Aligning reference translations with embedding similarity...")

    all_refs = []
    for text_id, info in BDK_PDFS.items():
        pdf_path = os.path.join(BDK_DIR, info["pdf"])
        if not os.path.exists(pdf_path):
            print(f"  SKIP {text_id}: PDF not found")
            continue

        chinese_sents = chinese_by_text.get(text_id, [])
        if not chinese_sents:
            print(f"  SKIP {text_id}: no Chinese sentences")
            continue

        print(f"  {text_id}: extracting from {info['pdf']}...")
        eng_text = extract_pdf_text(pdf_path, info["start_page"], info["end_page"])
        en_sents = clean_and_split_english(eng_text)

        zh_texts = [s["chinese"] for s in chinese_sents]
        print(f"    Chinese: {len(zh_texts)} sents, English: {len(en_sents)} sents")
        print(f"    Computing embeddings and aligning...")

        alignments = align_by_embedding(model, zh_texts, en_sents)
        print(f"    Aligned {len(alignments)} pairs (min_sim={MIN_SIMILARITY})")

        for zh_idx, en_idx, score in alignments:
            all_refs.append({
                "id": chinese_sents[zh_idx]["id"],
                "text_id": text_id,
                "chinese": chinese_sents[zh_idx]["chinese"],
                "english": en_sents[en_idx],
                "align_score": round(score, 3),
            })

        # Show a few examples
        for zh_idx, en_idx, score in alignments[:3]:
            print(f"      [{score:.2f}] {zh_texts[zh_idx][:40]}... -> {en_sents[en_idx][:50]}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for ref in all_refs:
            f.write(json.dumps(ref, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(all_refs)} aligned reference translations")
    print(f"Written to: {output_path}")


if __name__ == "__main__":
    main()
