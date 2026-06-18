# Introduction au Machine Learning

Le **Machine Learning** (apprentissage automatique) est une branche de l'intelligence artificielle qui permet aux systèmes d'apprendre à partir de données sans être explicitement programmés pour chaque tâche.

## Types d'apprentissage

### Apprentissage supervisé
L'apprentissage supervisé utilise des données étiquetées (features + labels). Exemples d'algorithmes :
- **Régression linéaire** : prédire une valeur continue (prix, température)
- **Régression logistique** : classification binaire
- **Arbres de décision** et **Random Forest** : modèles interprétables
- **Support Vector Machines (SVM)** : frontières de décision optimales
- **Réseaux de neurones** : modèles flexibles pour données complexes

### Apprentissage non supervisé
Sans labels, l'objectif est de découvrir des structures :
- **Clustering** (K-Means, DBSCAN) : regrouper des points similaires
- **Réduction de dimension** (PCA, t-SNE) : visualisation et compression
- **Détection d'anomalies** : identifier des comportements atypiques

### Apprentissage par renforcement
Un agent apprend par interaction avec un environnement via récompenses et pénalités. Applications : jeux (AlphaGo), robotique, recommandation.

## Pipeline ML typique

1. Collecte et nettoyage des données
2. Feature engineering et normalisation
3. Division train/validation/test (souvent 70/15/15)
4. Entraînement et sélection d'hyperparamètres
5. Évaluation (accuracy, F1, RMSE selon la tâche)
6. Déploiement et monitoring (drift detection)

## Métriques courantes

| Tâche | Métriques |
|-------|-----------|
| Classification | Accuracy, Precision, Recall, F1-score, AUC-ROC |
| Régression | MAE, MSE, RMSE, R² |
| Clustering | Silhouette score, Davies-Bouldin index |

## Overfitting et régularisation

L'**overfitting** survient quand le modèle mémorise les données d'entraînement. Solutions :
- Régularisation L1/L2
- Dropout (réseaux de neurones)
- Early stopping
- Augmentation de données
- Cross-validation
