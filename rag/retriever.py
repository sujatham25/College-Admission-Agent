"""
rag/retriever.py
----------------
Loads all .txt knowledge-base files, splits them into overlapping chunks,
builds the TF-IDF vector store, and exposes a retrieve() function used by
the admission agent.

DEMO PROJECT — IBM AICTE Internship 2026
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from rag.vector_store import Chunk, TFIDFVectorStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KB_DIR = Path(__file__).parent.parent / "knowledge_base"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_SCORE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Collapse multiple blank lines and strip leading/trailing whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(
    text: str,
    source: str,
    start_id: int
) -> List[Chunk]:
    """
    Split text into overlapping character-level chunks.
    """

    chunks: List[Chunk] = []
    chunk_id = start_id

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    current = ""

    for para in paragraphs:

        if len(current) + len(para) + 2 <= CHUNK_SIZE:

            current = (
                current + "\n\n" + para
            ).strip()

        else:

            if current:

                chunks.append(
                    Chunk(
                        text=current,
                        source=source,
                        chunk_id=chunk_id
                    )
                )

                chunk_id += 1

                current = (
                    current[-CHUNK_OVERLAP:].strip()
                    + "\n\n"
                    + para
                )

            else:

                lines = para.splitlines()

                for line in lines:

                    if len(current) + len(line) + 1 <= CHUNK_SIZE:

                        current = (
                            current + "\n" + line
                        ).strip()

                    else:

                        if current:

                            chunks.append(
                                Chunk(
                                    text=current,
                                    source=source,
                                    chunk_id=chunk_id
                                )
                            )

                            chunk_id += 1

                            current = (
                                current[-CHUNK_OVERLAP:].strip()
                                + "\n"
                                + line
                            )

                        else:

                            current = line

    if current:

        chunks.append(
            Chunk(
                text=current,
                source=source,
                chunk_id=chunk_id
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_knowledge_base(kb_dir: Path) -> List[Chunk]:
    """Read every .txt file in the knowledge base."""

    all_chunks: List[Chunk] = []
    chunk_id = 0

    if not kb_dir.exists():

        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_dir}\n"
            "Make sure the 'knowledge_base/' folder exists."
        )

    txt_files = sorted(
        kb_dir.glob("*.txt")
    )

    if not txt_files:

        raise FileNotFoundError(
            f"No .txt files found in {kb_dir}."
        )

    for filepath in txt_files:

        raw = filepath.read_text(
            encoding="utf-8"
        )

        clean = _clean_text(raw)

        source_label = filepath.name

        file_chunks = _chunk_text(
            clean,
            source=source_label,
            start_id=chunk_id
        )

        all_chunks.extend(file_chunks)

        chunk_id += len(file_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Module-level singleton store
# ---------------------------------------------------------------------------

_store: TFIDFVectorStore | None = None


def _get_store() -> TFIDFVectorStore:

    global _store

    if _store is None:

        chunks = _load_knowledge_base(KB_DIR)

        _store = TFIDFVectorStore()

        _store.add_chunks(chunks)

        _store.build()

    return _store


def reload_store() -> None:

    global _store

    _store = None

    _get_store()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 3
) -> List[Tuple[Chunk, float]]:
    """
    Retrieve relevant knowledge-base chunks using TF-IDF similarity
    with topic-based source priority.
    """

    store = _get_store()

    query_lower = query.lower()

    topic_sources: List[str] = []

    # Eligibility
    if any(word in query_lower for word in [
        "eligibility",
        "eligible",
        "qualification",
        "minimum marks",
        "percentage",
        "percent",
        "age",
        "jee main",
        "entrance exam"
    ]):

        topic_sources = [
            "eligibility.txt"
        ]

    # Documents
    elif any(word in query_lower for word in [
        "document",
        "documents",
        "certificate",
        "required papers"
    ]):

        topic_sources = [
            "documents.txt"
        ]

    # Fees
    elif any(word in query_lower for word in [
        "fee",
        "fees",
        "tuition",
        "cost",
        "application fee"
    ]):

        topic_sources = [
            "fees.txt",
            "faq.txt"
        ]

    # Deadlines
    elif any(word in query_lower for word in [
        "deadline",
        "deadlines",
        "last date",
        "important date"
    ]):

        topic_sources = [
            "deadlines.txt"
        ]

    # Scholarships
    elif any(word in query_lower for word in [
        "scholarship",
        "scholarships",
        "financial aid"
    ]):

        topic_sources = [
            "scholarships.txt"
        ]

    # Courses
    elif any(word in query_lower for word in [
        "course",
        "courses",
        "program",
        "programs",
        "branch",
        "branches"
    ]):

        topic_sources = [
            "courses.txt"
        ]

    # Application process
    elif any(word in query_lower for word in [
        "application process",
        "apply",
        "application",
        "admission process",
        "how to apply"
    ]):

        topic_sources = [
            "application_process.txt"
        ]


    # -----------------------------------------------------------------------
    # Retrieve ALL matching chunks first
    # -----------------------------------------------------------------------

    results = store.search_all(query)

    scored_results = []

    for chunk, score in results:

        boosted_score = score

        # Give directly relevant source a priority boost
        if chunk.source in topic_sources:

            boosted_score += 0.20

        scored_results.append(
            (
                chunk,
                boosted_score
            )
        )


    # Sort by relevance
    scored_results.sort(
        key=lambda item: item[1],
        reverse=True
    )


    # Return requested number of results
    final_results = []

    for chunk, boosted_score in scored_results:

        if boosted_score >= MIN_SCORE_THRESHOLD:

            final_results.append(
                (
                    chunk,
                    boosted_score
                )
            )

        if len(final_results) >= top_k:
            break

    return final_results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def get_chunk_count() -> int:

    return len(
        _get_store()._chunks
    )
