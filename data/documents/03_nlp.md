# Traitement du Langage Naturel (NLP)

Le **NLP** (Natural Language Processing) traite et génère du texte en langage humain.

## Étapes du pipeline NLP

1. **Tokenisation** : découper le texte en tokens (mots, sous-mots)
2. **Normalisation** : minuscules, suppression ponctuation
3. **Stemming/Lemmatisation** : réduire à la racine (courir → cour)
4. **Embedding** : représentation vectorielle dense du sens

## Représentations vectorielles

### Word2Vec (2013)
- **CBOW** : prédire un mot à partir du contexte
- **Skip-gram** : prédire le contexte à partir d'un mot
- Propriété : analogies (roi - homme + femme ≈ reine)

### GloVe
Co-occurrence matrix factorization, combine statistiques globales et locales.

### Embeddings contextuels
- **ELMo** : embeddings dépendant du contexte (BiLSTM)
- **BERT** : bidirectionnel, pré-entraîné sur MLM et NSP
- **Sentence-BERT** : embeddings de phrases entières

## Tâches NLP

| Tâche | Description |
|-------|-------------|
| Classification de texte | Sentiment, spam, catégorisation |
| NER | Extraction d'entités nommées (personnes, lieux) |
| QA | Question-réponse sur documents |
| Résumé | Abstractive ou extractive |
| Traduction | Seq2seq, Transformers |
| Génération | Complétion, dialogue |

## Modèles de langage (LLM)

Les **Large Language Models** sont entraînés sur des corpus massifs :
- Objectif : prédiction du token suivant (causal LM)
- Scaling laws : performance ∝ taille modèle × données × compute
- Exemples : GPT-4, Claude, Llama, Mistral, Gemini

### Prompting
- Zero-shot : instruction directe
- Few-shot : exemples dans le prompt
- Chain-of-Thought (CoT) : raisonnement étape par étape

### Fine-tuning
- **SFT** (Supervised Fine-Tuning) : données instruction-réponse
- **RLHF** : Reinforcement Learning from Human Feedback
- **LoRA/QLoRA** : fine-tuning efficace en paramètres

## Évaluation NLP

- **BLEU, ROUGE** : traduction, résumé
- **Perplexity** : qualité du modèle de langage
- **Benchmarks** : GLUE, SuperGLUE, MMLU, HumanEval
