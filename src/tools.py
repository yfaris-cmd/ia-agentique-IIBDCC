"""Outils disponibles pour l'agent RAG."""

from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStore

from src.config import RETRIEVAL_K
from src.document_loader import build_vector_store


def create_tools(vector_store: VectorStore | None = None):
    """Crée les outils liés à la base documentaire."""
    store = vector_store or build_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    @tool
    def search_documents(query: str) -> str:
        """Recherche sémantique dans la base documentaire sur l'Intelligence Artificielle.

        Utilisez cet outil pour trouver des informations sur : Machine Learning,
        Deep Learning, NLP, RAG, LangGraph, éthique de l'IA, etc.

        Args:
            query: Requête de recherche en langage naturel.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "Aucun document trouvé pour cette requête."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "inconnu")
            parts.append(f"[{i}] Source: {source}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @tool
    def list_document_topics() -> str:
        """Liste les thèmes couverts par la base documentaire."""
        return (
            "Thèmes disponibles dans la base :\n"
            "1. Machine Learning (supervisé, non supervisé, renforcement)\n"
            "2. Deep Learning (CNN, RNN, LSTM, Transformers)\n"
            "3. NLP (embeddings, LLM, prompting, fine-tuning)\n"
            "4. RAG (retrieval, chunking, Agentic RAG)\n"
            "5. LangGraph (state, nodes, edges, memory)\n"
            "6. Éthique et gouvernance de l'IA (biais, RGPD, AI Act)"
        )

    return [search_documents, list_document_topics]
