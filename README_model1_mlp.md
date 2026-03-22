# Model 1 — Optimized MLP with Hyperparameter Tuning

## Overview

This model improves upon Assignment 1's baseline MLP by:
- Expanding the feature representation (TF-IDF + sentiment lexicon scores)
- Applying batch normalization and dropout regularization
- Comparing multiple gradient-descent optimizers (Adam, SGD with momentum, RMSProp)
- Using early stopping and learning rate scheduling

**Baseline (Assignment 1) accuracy: 77.42%**
**Target: > 85%**

---

## Files

| File | Description |
|------|-------------|
| `model1_optimized_mlp.ipynb` | Training notebook with all steps |
| `results_model1_mlp.json` | Accuracy, AUC, classification report |
| `model1_optimized_mlp.keras` | Saved Keras model |
| `optimizer_comparison.png` | Validation accuracy by optimizer |
| `mlp_training_curves.png` | Loss & accuracy curves |
| `mlp_confusion_matrix.png` | Test confusion matrix |
| `mlp_roc_curve.png` | ROC curve |

---

## Model Architecture

```
Input (5005 features)
    │
    ▼
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
Dense(1)   → Sigmoid
```

**Total parameters**: ~1.4M

---

## Feature Engineering

Two feature sets are combined into a 5005-dimensional vector:

### 1. TF-IDF Features (5000 dims)
- `TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=5)`
- Captures unigrams and bigrams from the cleaned review text
- `min_df=5` removes very rare terms (noise reduction)

### 2. Sentiment Lexicon Features (5 dims)
Same as Assignment 1:
- VADER negative score
- VADER neutral score
- VADER positive score
- VADER compound score
- TextBlob polarity

### Why TF-IDF helps
Assignment 1 only used 5 hand-crafted features. These aggregate all sentiment
into a single score, losing word-level information. TF-IDF gives the model direct
access to the vocabulary, allowing it to learn which specific words/phrases
signal positive or negative sentiment.

---

## Optimization Techniques Applied

### 1. Mini-Batch Gradient Descent
- `batch_size=256` — balances speed and gradient stability
- Mini-batches introduce noise that acts as implicit regularization

### 2. Adam Optimizer
- Combines Momentum (EWMA of gradients) and RMSProp (EWMA of squared gradients)
- Adaptive per-parameter learning rates
- Usually fastest convergence for this type of problem
- `learning_rate=0.001` (default)

### 3. SGD with Nesterov Momentum
- Nesterov variant looks ahead before computing gradient
- `momentum=0.9` — accumulates past gradient directions
- Good when well-tuned; compared against Adam in this notebook

### 4. RMSProp
- Divides learning rate by EWMA of squared gradients
- Adapts lr per parameter; good for non-stationary problems

### 5. Batch Normalization
- Applied after every Dense layer, before activation
- Normalizes layer inputs to have zero mean and unit variance
- Benefits: faster training, higher learning rates, slight regularization effect

### 6. Dropout Regularization
- `rate=0.3` → 30% of neurons randomly zeroed each forward pass during training
- Prevents co-adaptation of neurons → reduces overfitting

### 7. Early Stopping
- Monitors `val_loss`; stops after 5 epochs without improvement
- Restores best weights automatically

### 8. ReduceLROnPlateau
- Halves learning rate when `val_loss` plateaus for 3 consecutive epochs
- Prevents the optimizer from overshooting optimal weights

### 9. Feature Scaling
- `StandardScaler` applied to combined features
- Zero-mean, unit-variance normalization → essential for gradient descent convergence

---

## Hyperparameter Tuning

The following hyperparameters were explored:

| Hyperparameter | Values Tried | Best |
|----------------|--------------|------|
| Hidden layers | (64,32), (128,64), (256,128,64) | (256,128,64) |
| Optimizer | Adam, SGD, RMSProp | Adam |
| Dropout rate | 0.2, 0.3, 0.5 | 0.3 |
| Batch size | 64, 128, 256 | 256 |
| Learning rate | 1e-4, 1e-3, 5e-3 | 1e-3 |

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | See `results_model1_mlp.json` |
| ROC-AUC | See `results_model1_mlp.json` |
| Best Optimizer | Adam |

**Improvement over Assignment 1 baseline (77.42%):** See results file.

---

## How to Run

1. Open `model1_optimized_mlp.ipynb` in Google Colab
2. Run all cells in order (top → bottom)
3. Results will be saved automatically as JSON and PNG files

### Requirements
```
tensorflow>=2.13
scikit-learn>=1.3
nltk
textblob
vaderSentiment
kagglehub
scipy
```
