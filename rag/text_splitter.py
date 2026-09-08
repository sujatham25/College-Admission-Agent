# rag/text_splitter.py
# Splits large documents into smaller, overlapping chunks so that
# each chunk fits within the embedding model's context window and
# carries enough surrounding context for meaningful retrieval.

from typing import List, Dict


class TextSplitter:
    """Splits text documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size:    Maximum number of characters per chunk.
            chunk_overlap: Number of characters that overlap between consecutive chunks.
                           This preserves context across chunk boundaries.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split a list of documents into chunks.

        Args:
            documents: List of dicts with 'source' and 'content' keys.

        Returns:
            List of chunk dicts, each with:
                'source'     — original filename
                'content'    — chunk text
                'chunk_index'— integer index of this chunk within the source document
        """
        chunks: List[Dict[str, str]] = []

        for doc in documents:
            source = doc["source"]
            text = doc["content"]
            doc_chunks = self._split_text(text)

            for idx, chunk_text in enumerate(doc_chunks):
                chunks.append(
                    {
                        "source": source,
                        "content": chunk_text,
                        "chunk_index": idx,
                    }
                )

        print(f"[TextSplitter] Created {len(chunks)} chunk(s) from {len(documents)} document(s).")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Split a single text string into overlapping character-based chunks.
        Attempts to split on paragraph / line boundaries when possible.
        """
        # Prefer splitting on double-newlines (paragraph boundaries) first.
        # Then fall back to a hard character split if no good boundary is found.
        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end >= text_length:
                # Last chunk: take everything that remains
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to find a paragraph break (double newline) near the end of the window
            split_at = self._find_split_boundary(text, start, end)
            chunk = text[start:split_at].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward by (chunk_size - overlap) but at least 1 char
            step = max(split_at - start - self.chunk_overlap, 1)
            start += step

        return chunks

    def _find_split_boundary(self, text: str, start: int, end: int) -> int:
        """
        Look for a paragraph break (\\n\\n) or single newline close to `end`.
        If none found, return `end` for a hard cut.
        """
        search_window = text[start:end]

        # Prefer double newline
        pos = search_window.rfind("\n\n")
        if pos != -1 and pos > self.chunk_size // 2:
            return start + pos + 2  # include the newlines in the previous chunk

        # Fall back to single newline
        pos = search_window.rfind("\n")
        if pos != -1 and pos > self.chunk_size // 2:
            return start + pos + 1

        return end
