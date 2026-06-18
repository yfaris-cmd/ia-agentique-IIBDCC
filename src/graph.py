"""Architecture LangGraph de l'agent RAG agentique."""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.document_loader import get_retriever
from src.models import get_llm
from src.state import AgentState

MAX_RETRIES = 2

# --- Schémas structurés pour le routage et le grading ---


class RouteQuery(BaseModel):
    """Décision de routage : recherche documentaire ou réponse directe."""

    datasource: Literal["vectorstore", "direct"] = Field(
        description="vectorstore si la question nécessite la base documentaire, "
        "direct pour salutations ou connaissances générales sans source."
    )


class GradeDocuments(BaseModel):
    """Évaluation binaire de pertinence d'un document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' si le document est pertinent pour la question, 'no' sinon."
    )


class GradeGeneration(BaseModel):
    """Vérification que la réponse est ancrée dans les documents."""

    grounded: Literal["yes", "no"] = Field(
        description="'yes' si la réponse est supportée par les documents."
    )
    useful: Literal["yes", "no"] = Field(
        description="'yes' si la réponse répond utilement à la question."
    )


# --- Prompts ---

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un routeur pour un assistant IA spécialisé en Intelligence Artificielle. "
            "Décide si la question nécessite une recherche dans la base documentaire (vectorstore) "
            "ou peut être traitée directement (direct). "
            "Utilise vectorstore pour toute question technique sur ML, DL, NLP, RAG, LangGraph, éthique IA. "
            "Utilise direct uniquement pour salutations ou questions hors sujet.",
        ),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reformule la question pour améliorer la recherche documentaire. "
            "Garde le sens original, ajoute des termes techniques pertinents si nécessaire.",
        ),
        ("human", "Question originale : {question}\n\nQuestion reformulée :"),
    ]
)

GRADE_DOC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Évalue si le document est pertinent pour répondre à la question. "
            "Réponds yes ou no.",
        ),
        ("human", "Document :\n{document}\n\nQuestion : {question}"),
    ]
)

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un expert en Intelligence Artificielle. "
            "Réponds à la question UNIQUEMENT à partir du contexte fourni. "
            "Si le contexte est insuffisant, dis-le clairement. "
            "Cite les sources quand c'est possible. Réponds en français.",
        ),
        (
            "human",
            "Contexte :\n{context}\n\nQuestion : {question}\n\nRéponse :",
        ),
    ]
)

DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant IA spécialisé en Intelligence Artificielle. "
            "Réponds brièvement en français. "
            "Pour les questions techniques détaillées, invite l'utilisateur à poser une question précise.",
        ),
        ("human", "{question}"),
    ]
)


def build_graph(vector_store: VectorStore | None = None):
    """Construit et compile le graphe LangGraph avec mémoire."""
    llm = get_llm()
    retriever = get_retriever(vector_store)

    structured_router = llm.with_structured_output(RouteQuery)
    structured_grader = llm.with_structured_output(GradeDocuments)
    structured_gen_grader = llm.with_structured_output(GradeGeneration)

    # --- Nœuds ---

    def route_question(state: AgentState) -> dict:
        question = state["question"]
        result = structured_router.invoke(ROUTER_PROMPT.format_messages(question=question))
        return {"route": result.datasource}

    def retrieve(state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        docs = retriever.invoke(query)
        sources = [
            doc.metadata.get("source", "inconnu").split("\\")[-1].split("/")[-1]
            for doc in docs
        ]
        return {"documents": docs, "sources": sources}

    def grade_documents(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state["question"]
        relevant = []
        for doc in state.get("documents", []):
            grade = structured_grader.invoke(
                GRADE_DOC_PROMPT.format_messages(
                    document=doc.page_content[:1500],
                    question=question,
                )
            )
            if grade.binary_score == "yes":
                relevant.append(doc)
        return {"graded_documents": relevant}

    def rewrite_question(state: AgentState) -> dict:
        question = state["question"]
        response = llm.invoke(REWRITE_PROMPT.format_messages(question=question))
        retry = state.get("retry_count", 0) + 1
        return {
            "rewritten_question": response.content.strip(),
            "retry_count": retry,
            "documents": [],
            "graded_documents": [],
        }

    def generate(state: AgentState) -> dict:
        question = state.get("rewritten_question") or state["question"]
        docs = state.get("graded_documents") or state.get("documents", [])
        context = "\n\n".join(doc.page_content for doc in docs)
        response = llm.invoke(
            GENERATE_PROMPT.format_messages(context=context, question=question)
        )
        return {"generation": response.content}

    def generate_direct(state: AgentState) -> dict:
        response = llm.invoke(DIRECT_PROMPT.format_messages(question=state["question"]))
        return {"generation": response.content}

    def finalize(state: AgentState) -> dict:
        answer = state.get("generation", "")
        sources = list(dict.fromkeys(state.get("sources", [])))
        if sources:
            answer += f"\n\nSources consultées : {', '.join(sources)}"
        return {
            "messages": [AIMessage(content=answer)],
            "generation": answer,
        }

    # --- Routage conditionnel ---

    def after_route(state: AgentState) -> str:
        return "retrieve" if state["route"] == "vectorstore" else "generate_direct"

    def after_grade(state: AgentState) -> str:
        if state.get("graded_documents"):
            return "generate"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "generate"
        return "rewrite"

    def after_generate(state: AgentState) -> str:
        question = state.get("rewritten_question") or state["question"]
        docs = state.get("graded_documents") or state.get("documents", [])
        if not docs:
            return "finalize"
        context = "\n\n".join(doc.page_content[:800] for doc in docs)
        grade = structured_gen_grader.invoke(
            [
                SystemMessage(
                    content="Évalue si la réponse est ancrée dans les faits et utile."
                ),
                HumanMessage(
                    content=f"Documents :\n{context}\n\nQuestion : {question}\n\n"
                    f"Réponse : {state.get('generation', '')}"
                ),
            ]
        )
        if grade.grounded == "yes" and grade.useful == "yes":
            return "finalize"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "finalize"
        return "rewrite"

    # --- Assemblage du graphe ---

    workflow = StateGraph(AgentState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_direct", generate_direct)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "route_question")
    workflow.add_conditional_edges("route_question", after_route)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", after_grade)
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_conditional_edges("generate", after_generate)
    workflow.add_edge("generate_direct", "finalize")
    workflow.add_edge("finalize", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def ask(question: str, thread_id: str = "default") -> dict:
    """Point d'entrée simplifié pour poser une question à l'agent."""
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "retry_count": 0,
            "sources": [],
        },
        config,
    )
    return result
