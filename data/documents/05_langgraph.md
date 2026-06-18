# LangGraph — Orchestration d'agents

**LangGraph** est une bibliothèque de LangChain pour construire des agents stateful sous forme de graphes.

## Concepts fondamentaux

### State (État)
État partagé entre tous les nœuds du graphe. Défini via `TypedDict` ou `Pydantic` :
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list[str]
```

### Nodes (Nœuds)
Fonctions qui reçoivent l'état et retournent des mises à jour :
```python
def retrieve(state: AgentState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}
```

### Edges (Arêtes)
- **Normal edges** : transition fixe A → B
- **Conditional edges** : routage dynamique selon l'état
```python
graph.add_conditional_edges("grade", route_documents, {
    "generate": "generate",
    "rewrite": "rewrite_question"
})
```

### Memory (Mémoire)
- **Checkpointer** : sauvegarde l'état entre invocations (MemorySaver, SqliteSaver)
- Permet conversations multi-tours avec `thread_id`
- Human-in-the-loop : interruption et reprise

## Différence avec create_agent

| create_agent (LangChain) | LangGraph manuel |
|--------------------------|------------------|
| Boucle ReAct pré-construite | Contrôle total du flux |
| Peu personnalisable | Nœuds et arêtes custom |
| Rapide à prototyper | Architecture explicite |
| Boîte noire | Visualisable (Mermaid) |

## Patterns Agentic RAG avec LangGraph

### Adaptive RAG
1. Router : la question nécessite-t-elle une recherche ?
2. Retrieve : recherche vectorielle
3. Grade : documents pertinents ?
4. Generate ou Rewrite + boucle

### Self-RAG
Le modèle génère, puis auto-évalue (retrieval needed? relevant? supported?)

### Plan-and-Execute
Planificateur décompose → exécuteur par étape → synthèse

## Compilation et exécution

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_edge(START, "retrieve")
# ...
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session-1"}}
app.invoke({"question": "..."}, config)
```

## Visualisation

```python
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

Ou export Mermaid pour documentation.

## Bonnes pratiques

- Garder les nœuds petits et à responsabilité unique
- Logger les transitions pour le debug
- Limiter les boucles (max_iterations) pour éviter cycles infinis
- Typer strictement l'état pour la maintenabilité
