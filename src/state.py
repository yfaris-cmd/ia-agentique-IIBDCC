"""Définition de l'état partagé du graphe LangGraph."""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """État traversant tous les nœuds de l'agent RAG."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    rewritten_question: str
    documents: list
    graded_documents: list
    generation: str
    route: Literal["vectorstore", "direct", "rewrite"]
    retry_count: int
    sources: Annotated[list[str], operator.add]
