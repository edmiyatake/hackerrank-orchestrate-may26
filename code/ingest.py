"""
ingest.py — Phase 1: Corpus ingestion, chunking, and BM25 index builder.

Usage:
    python ingest.py --data-dir ../data --index-out index.pkl

Produces a pickle file containing:
    {
        "chunks": List[dict],   # all chunk records
        "bm25":   BM25Okapi,   # fitted index (parallel to chunks)
    }

Each chunk record:
    {
        "text":          str,   # raw chunk text (used for BM25 + LLM context)
        "domain":        str,   # "claude" | "hackerrank" | "visa"
        "product_area":  str,   # immediate parent folder name
        "article_title": str,   # human-readable title (numeric prefix stripped)
        "source_path":   str,   # relative path from data-dir root
    }
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import tiktoken
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_CHUNK = 300
CHUNK_OVERLAP_TOKENS = 50          # sliding overlap when splitting large sections
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_len(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _clean_title(stem: str) -> str:
    """Strip leading numeric ID and convert hyphens to spaces.

    '7996918-what-is-amazon-bedrock' → 'what is amazon bedrock'
    'traveler's checks'              → 'traveler's checks'
    """
    cleaned = re.sub(r"^\d+-", "", stem)
    return cleaned.replace("-", " ")


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown on heading lines (# / ## / ###).

    Returns list of (heading, body) pairs.
    The first pair may have an empty heading if content precedes any heading.
    """
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    matches = list(heading_re.finditer(text))

    # Content before the first heading
    first_start = matches[0].start() if matches else len(text)
    preamble = text[:first_start].strip()
    if preamble:
        sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if heading or body:
            sections.append((heading, body))

    return sections


def _chunk_text(heading: str, body: str, max_tokens: int, overlap: int) -> list[str]:
    """Turn a (heading, body) section into ≤max_tokens chunks with overlap.

    Each chunk is prefixed with the heading so BM25 sees the topic signal.
    """
    prefix = f"{heading}\n" if heading else ""
    full = f"{prefix}{body}".strip()

    if _token_len(full) <= max_tokens:
        return [full] if full else []

    # Sentence-aware split: prefer breaking on '. ' or '\n'
    tokens = TOKENIZER.encode(full)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - overlap

    return chunks


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def ingest_file(path: Path, data_dir: Path) -> list[dict]:
    """Parse one Markdown file → list of chunk dicts."""
    relative = path.relative_to(data_dir)
    parts = relative.parts  # e.g. ('claude', 'amazon-bedrock', 'file.md')

    domain = parts[0]                         # claude / hackerrank / visa
    product_area = path.parent.name           # immediate parent folder
    article_title = _clean_title(path.stem)
    source_path = str(relative)

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  [WARN] Could not read {path}: {exc}", file=sys.stderr)
        return []

    sections = _split_into_sections(raw)
    chunks: list[dict] = []

    for heading, body in sections:
        for chunk_text in _chunk_text(heading, body, MAX_TOKENS_PER_CHUNK, CHUNK_OVERLAP_TOKENS):
            chunks.append(
                {
                    "text": chunk_text,
                    "domain": domain,
                    "product_area": product_area,
                    "article_title": article_title,
                    "source_path": source_path,
                }
            )

    return chunks


def ingest_corpus(data_dir: Path) -> list[dict]:
    """Walk data_dir recursively, ingest every .md file."""
    all_chunks: list[dict] = []
    md_files = sorted(data_dir.rglob("*.md"))

    if not md_files:
        print(f"[ERROR] No .md files found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(md_files)} markdown files — ingesting...")
    for path in md_files:
        chunks = ingest_file(path, data_dir)
        all_chunks.extend(chunks)
        print(f"  {path.relative_to(data_dir)}  →  {len(chunks)} chunk(s)")

    return all_chunks


# ---------------------------------------------------------------------------
# BM25 index builder
# ---------------------------------------------------------------------------

def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Tokenize chunk texts and fit a BM25Okapi index."""
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest support corpus and build BM25 index.")
    p.add_argument("--data-dir", type=Path, default=Path("../data"),
                   help="Root of the data/ directory (default: ../data)")
    p.add_argument("--index-out", type=Path, default=Path("index.pkl"),
                   help="Output path for the serialised index (default: index.pkl)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"[ERROR] data-dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Ingest all Markdown files
    chunks = ingest_corpus(data_dir)
    print(f"\nTotal chunks: {len(chunks)}")

    # Domain breakdown
    from collections import Counter
    domain_counts = Counter(c["domain"] for c in chunks)
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} chunks")

    # 2. Build BM25 index
    print("\nBuilding BM25 index...")
    bm25 = build_bm25(chunks)

    # 3. Serialize to disk
    index_out = args.index_out.resolve()
    payload = {"chunks": chunks, "bm25": bm25}
    with open(index_out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Index written to: {index_out}")
    print("Done.")


if __name__ == "__main__":
    main()