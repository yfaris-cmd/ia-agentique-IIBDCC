# Deep Learning et Réseaux de Neurones

Le **Deep Learning** utilise des réseaux de neurones profonds (plusieurs couches cachées) pour modéliser des représentations hiérarchiques des données.

## Perceptron et neurone artificiel

Un neurone calcule : `output = activation(Σ(w_i * x_i) + b)`

Fonctions d'activation courantes :
- **ReLU** : max(0, x) — la plus utilisée dans les couches cachées
- **Sigmoid** : sortie entre 0 et 1 — classification binaire
- **Softmax** : distribution de probabilités — classification multi-classe
- **Tanh** : sortie entre -1 et 1

## Architectures majeures

### CNN (Convolutional Neural Networks)
Conçus pour les données spatialles (images). Composants :
- Couches convolutives : détection de motifs locaux
- Pooling (max/average) : réduction dimensionnelle
- Applications : classification d'images, segmentation, détection d'objets

### RNN et LSTM
Réseaux récurrents pour séquences temporelles :
- **RNN** : mémoire à court terme, problème du gradient vanishing
- **LSTM** : portes (forget, input, output) pour mémoire long terme
- **GRU** : variante simplifiée de LSTM
- Applications : traduction, séries temporelles, reconnaissance vocale

### Transformers
Architecture basée sur l'**attention** (Vaswani et al., 2017) :
- **Self-attention** : chaque token attend à tous les autres
- **Multi-head attention** : plusieurs projections parallèles
- **Positional encoding** : information de position
- Modèles : BERT (encodeur), GPT (décodeur), T5 (encodeur-décodeur)

## Entraînement

- **Backpropagation** : calcul des gradients par chaîne de dérivation
- **Optimiseurs** : SGD, Adam, AdamW, RMSprop
- **Batch normalization** : stabilise l'entraînement
- **Learning rate scheduling** : warmup, cosine decay

## Frameworks

- **PyTorch** : recherche, dynamic computation graph
- **TensorFlow/Keras** : production, TensorFlow Serving
- **JAX** : autodiff fonctionnelle, accélération GPU/TPU

## Transfer Learning

Réutiliser un modèle pré-entraîné sur de grandes quantités de données :
1. Fine-tuning complet
2. Feature extraction (gel des couches basses)
3. Adaptation avec peu de données (few-shot)

Exemples : ImageNet pour vision, BERT/GPT pour NLP.
