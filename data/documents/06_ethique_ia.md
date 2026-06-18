# Éthique et gouvernance de l'IA

L'**éthique de l'IA** couvre les enjeux moraux, sociaux et juridiques du déploiement des systèmes intelligents.

## Principes fondamentaux

### Transparence et explicabilité
- Comprendre comment une décision est prise (XAI)
- Documentation des modèles (model cards, datasheets)
- Communication claire aux utilisateurs sur les limites

### Équité et biais
Sources de biais :
- Données d'entraînement non représentatives
- Features corrélées à des attributs sensibles
- Feedback loops amplifiant les inégalités

Mitigation :
- Audit des datasets (demographic parity, equalized odds)
- Débiaisage pré/post-entraînement
- Diversité dans les équipes de développement

### Vie privée
- **RGPD** (Europe) : consentement, droit à l'oubli, minimisation
- **Anonymisation** et pseudonymisation
- **Federated Learning** : entraînement décentralisé
- **Differential Privacy** : bruit calibré pour protéger les individus

### Sécurité et robustesse
- **Adversarial attacks** : perturbations imperceptibles
- **Prompt injection** : manipulation des LLM
- **Data poisoning** : corruption des données d'entraînement
- Red teaming et évaluations de sécurité

## Réglementation

### UE — AI Act (2024)
Classification par niveau de risque :
- **Inacceptable** : scoring social, manipulation subliminale
- **Haut risque** : recrutement, crédit, justice — obligations strictes
- **Risque limité** : chatbots — transparence
- **Risque minimal** : jeux, filtres spam

### Autres cadres
- NIST AI RMF (USA)
- ISO/IEC 42001 — management systems for AI
- UNESCO Recommendation on AI Ethics

## IA générative — risques spécifiques

| Risque | Description | Mitigation |
|--------|-------------|------------|
| Hallucinations | Faits inventés | RAG, grounding, vérification |
| Deepfakes | Contenu synthétique trompeur | Watermarking, détection |
| Propriété intellectuelle | Reproduction de contenu protégé | Filtrage, licensing |
| Impact environnemental | Consommation énergétique | Modèles efficaces, green AI |

## Gouvernance organisationnelle

- Comité éthique IA
- Impact assessments (AIA, DPIA)
- Monitoring post-déploiement
- Procédures de recours pour les personnes affectées
- Formation des équipes aux risques

## Responsible AI en pratique

Checklist avant déploiement :
1. Objectif légitime et proportionné ?
2. Données collectées légalement et de qualité ?
3. Biais mesurés et acceptables ?
4. Explicabilité suffisante pour le contexte ?
5. Plan de monitoring et de désactivation ?
