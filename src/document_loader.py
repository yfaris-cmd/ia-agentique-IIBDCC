"""Construction et chargement de la base documentaire vectorielle."""

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    RETRIEVAL_K,
    VECTOR_STORE_DIR,
)
from src.models import get_embeddings


def load_raw_documents(documents_dir: Path | None = None):
    """Charge les fichiers Markdown du répertoire documents."""
    path = documents_dir or DOCUMENTS_DIR
    loader = DirectoryLoader(
        str(path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    return loader.load()


def split_documents(documents):
    """Découpe les documents en chunks pour l'indexation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


def build_vector_store(force_rebuild: bool = False) -> VectorStore:
    """Construit ou charge la base vectorielle Chroma."""
    embeddings = get_embeddings()

    if VECTOR_STORE_DIR.exists() and not force_rebuild:
        return Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    raw_docs = load_raw_documents()
    chunks = split_documents(raw_docs)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    return vector_store


def get_retriever(vector_store: VectorStore | None = None, k: int | None = None):
    """Retourne un retriever configuré sur la base documentaire."""
    store = vector_store or build_vector_store()
    return store.as_retriever(search_kwargs={"k": k or RETRIEVAL_K})
