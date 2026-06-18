"""Configuration et instanciation des modèles LLM et embeddings."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """LLM principal pour génération et raisonnement."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY manquante. Copiez .env.example vers .env et renseignez votre clé."
        )
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Modèle d'embeddings pour la recherche vectorielle."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY manquante.")
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
