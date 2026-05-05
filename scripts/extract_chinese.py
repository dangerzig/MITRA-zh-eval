#!/usr/bin/env python3
"""
Extract sentence-level Chinese text from CBETA XML files.

Replicates the MITRA-zh-eval methodology: take the first 50-100 sentences
after prefaces and introductory paragraphs from each text. Clean line-end
hyphenations, line returns, and note anchors.

Usage:
    python3 extract_chinese.py --xml-dir ~/taisho-canon/xml/T \
        --output ~/MITRA-zh-eval/data/chinese_sentences.jsonl
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

TEI_NS = "http://www.tei-c.org/ns/1.0"
CB_NS = "http://www.cbeta.org/ns/1.0"

# The 32 texts from MITRA-zh-eval Table 1
EVAL_TEXTS = [
    ("T01", "0001"), ("T02", "0099"), ("T02", "0142"), ("T04", "0198"),
    ("T08", "0246"), ("T09", "0273"), ("T11", "0316"), ("T12", "0374"),
    ("T13", "0417"), ("T14", "0450"), ("T14", "0515"), ("T17", "0842"),
    ("T19", "0959"), ("T19", "1022B"), ("T20", "1060"), ("T20", "1077"),
    ("T20", "1136"), ("T20", "1166"), ("T21", "1261"), ("T21", "1277"),
    ("T21", "1305"), ("T21", "1394"), ("T24", "1492"), ("T30", "1568"),
    ("T32", "1666"), ("T34", "1725"), ("T37", "1762"), ("T42", "1826"),
    ("T45", "1857"), ("T45", "1909"), ("T47", "1961"), ("T48", "2004"),
]

# Target sentences per text (paper says "first 50-100 sentences")
TARGET_SENTENCES = 80


def extract_text_from_element(elem):
    """Recursively extract text from a TEI element, skipping notes."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        tag = child.tag.replace(f"{{{TEI_NS}}}", "").replace(f"{{{CB_NS}}}", "")
        # Skip notes, anchors, and apparatus
        if tag in ("note", "anchor", "app", "ref"):
            pass
        elif tag == "rdg":
            pass
        elif tag == "lem":
            parts.append(extract_text_from_element(child))
        elif tag == "g":
            # CBETA gaiji - try to get the ref character
            ref = child.get("ref", "")
            if child.text:
                parts.append(child.text)
            elif ref:
                parts.append(f"[{ref}]")
        else:
            parts.append(extract_text_from_element(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def find_xml_files(xml_dir, vol, num):
    """Find all XML files for a given text number."""
    vol_dir = os.path.join(xml_dir, vol)
    if not os.path.isdir(vol_dir):
        return []
    pattern = f"{vol}n{num}"
    files = []
    for f in sorted(os.listdir(vol_dir)):
        if f.startswith(pattern) and f.endswith(".xml"):
            files.append(os.path.join(vol_dir, f))
    return files


def extract_sentences(xml_files, target=TARGET_SENTENCES):
    """Extract Chinese sentences from CBETA XML files.

    Skips the first few paragraphs (preface material) then collects
    sentences by splitting on Chinese period (。), exclamation (！),
    and question mark (？).
    """
    all_text = []

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()

        # Find all <p> and <lg> (verse) elements in the body
        for elem in root.iter():
            tag = elem.tag.replace(f"{{{TEI_NS}}}", "")
            if tag in ("p", "lg"):
                text = extract_text_from_element(elem)
                text = text.strip()
                if text:
                    all_text.append(text)

    # Join all paragraphs
    full_text = "\n".join(all_text)

    # Clean up
    full_text = re.sub(r"\s+", "", full_text)  # Remove all whitespace
    full_text = re.sub(r"\[CB\d+\]", "", full_text)  # Remove CBETA refs
    full_text = re.sub(r"\[.*?\]", "", full_text)  # Remove bracketed refs

    # Split into sentences on Chinese punctuation
    raw_sentences = re.split(r"(?<=[。！？])", full_text)
    raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

    # Merge short sentences with the next one to avoid trivially easy
    # fragments. Nehrdich used Bertalign which naturally merges these.
    # Target minimum ~20 chars per sentence (roughly one full clause).
    MIN_CHARS = 20
    merged = []
    buf = ""
    for s in raw_sentences:
        buf += s
        if len(buf) >= MIN_CHARS:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += buf
        else:
            merged.append(buf)

    # Skip first ~5 sentences (title/preface material)
    skip = min(5, len(merged) // 10)
    merged = merged[skip:]

    # Take target number
    return merged[:target]


def main():
    parser = argparse.ArgumentParser(
        description="Extract Chinese sentences from CBETA XML for evaluation."
    )
    parser.add_argument(
        "--xml-dir", type=str, default=os.path.expanduser("~/taisho-canon/xml/T"),
        help="Path to CBETA XML T directory.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=os.path.expanduser("~/MITRA-zh-eval/data/chinese_sentences.jsonl"),
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_SENTENCES,
        help="Target sentences per text.",
    )
    args = parser.parse_args()

    results = []
    for vol, num in EVAL_TEXTS:
        text_id = f"{vol}n{num}"
        xml_files = find_xml_files(args.xml_dir, vol, num)
        if not xml_files:
            print(f"  SKIP {text_id}: no XML files found")
            continue

        sentences = extract_sentences(xml_files, args.target)
        print(f"  {text_id}: {len(sentences)} sentences from {len(xml_files)} file(s)")

        for i, sent in enumerate(sentences):
            results.append({
                "id": f"{text_id}_{i:04d}",
                "text_id": text_id,
                "sentence_idx": i,
                "chinese": sent,
            })

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(results)} sentences from {len(EVAL_TEXTS)} texts")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
