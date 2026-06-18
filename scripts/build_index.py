"""Construit la base vectorielle à partir des documents."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import build_vector_store, load_raw_documents, split_documents


def main(force: bool = False):
    print("Chargement des documents...")
    raw = load_raw_documents()
    print(f"  {len(raw)} fichiers chargés")

    chunks = split_documents(raw)
    print(f"  {len(chunks)} chunks créés")

    print("Indexation vectorielle (Chroma)...")
    build_vector_store(force_rebuild=force)
    print("Base vectorielle prête.")


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    main(force=force_rebuild)
