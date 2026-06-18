# RAG — Retrieval-Augmented Generation

Le **RAG** (Retrieval-Augmented Generation) combine recherche documentaire et génération par LLM pour produire des réponses ancrées dans des sources fiables.

## Architecture RAG classique

```
Question → Embedding → Recherche vectorielle → Documents pertinents
                                                      ↓
Réponse ← LLM ← Prompt (question + contexte)
```

## Composants

### 1. Indexation
- Chargement des documents (PDF, Markdown, HTML, API)
- **Chunking** : découpage en morceaux (500-1000 tokens)
- **Embedding** : conversion en vecteurs (OpenAI, HuggingFace, Cohere)
- **Stockage** : Chroma, Pinecone, FAISS, Weaviate, Qdrant

### 2. Récupération (Retrieval)
- **Similarité cosinus** : mesure standard entre embeddings
- **Top-k** : retourner les k chunks les plus similaires
- **MMR** (Maximal Marginal Relevance) : diversifier les résultats
- **Hybrid search** : combiner recherche vectorielle + BM25 (lexicale)

### 3. Génération
- Prompt incluant contexte récupéré + question
- Instructions : "Réponds uniquement à partir du contexte"
- Citations des sources utilisées

## Stratégies de chunking

| Stratégie | Avantages | Inconvénients |
|-----------|-----------|---------------|
| Fixed-size | Simple | Coupe au milieu des phrases |
| Recursive | Respecte la structure | Paramètres à ajuster |
| Semantic | Chunks cohérents sémantiquement | Plus coûteux |
| Parent-child | Contexte large + précision fine | Complexité |

## Limites du RAG naïf

- Récupération de documents non pertinents
- Contexte insuffisant pour questions complexes
- Pas de raisonnement multi-étapes
- Hallucinations malgré le contexte

## Agentic RAG

L'**Agentic RAG** ajoute un agent avec capacités de :
- **Planification** : décomposer la question en sous-questions
- **Routing** : choisir la bonne source ou outil
- **Grading** : évaluer la pertinence des documents récupérés
- **Self-correction** : reformuler la requête si résultats insuffisants
- **Tool use** : calculatrice, recherche web, API externes

Implémenté typiquement avec **LangGraph** : graphe d'états avec nœuds conditionnels et boucles.

## Métriques d'évaluation RAG

- **Faithfulness** : la réponse est-elle supportée par les sources ?
- **Answer relevance** : la réponse répond-elle à la question ?
- **Context precision/recall** : qualité de la récupération
- **Latency** : temps de réponse end-to-end
