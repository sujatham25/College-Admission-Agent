# rag/document_loader.py
# Loads plain-text documents from the knowledge_base directory.
# Each file is returned as a dict with 'source' (filename) and 'content' (text).

import os
from typing import List, Dict


class DocumentLoader:
    """Loads text documents from a specified directory."""

    def __init__(self, knowledge_base_dir: str):
        """
        Args:
            knowledge_base_dir: Path to the folder containing .txt knowledge base files.
        """
        self.knowledge_base_dir = knowledge_base_dir

    def load(self) -> List[Dict[str, str]]:
        """
        Read all .txt files in the knowledge_base directory.

        Returns:
            A list of dicts, each with keys:
                'source'  — filename (e.g. 'courses.txt')
                'content' — full text of the file
        """
        documents: List[Dict[str, str]] = []

        if not os.path.isdir(self.knowledge_base_dir):
            raise FileNotFoundError(
                f"Knowledge base directory not found: {self.knowledge_base_dir}"
            )

        for filename in sorted(os.listdir(self.knowledge_base_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_base_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        documents.append({"source": filename, "content": content})
                except Exception as e:
                    print(f"[DocumentLoader] Warning: Could not read '{filename}': {e}")

        if not documents:
            raise ValueError(
                f"No .txt documents found in: {self.knowledge_base_dir}"
            )

        print(f"[DocumentLoader] Loaded {len(documents)} document(s).")
        return documents
