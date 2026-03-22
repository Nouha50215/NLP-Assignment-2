# Model 3 — 1D Convolutional Neural Networks (TextCNN)

## Overview

1D CNNs apply learned filters (kernels) across a token sequence to detect
local patterns — equivalent to learning weighted n-gram detectors. They are
significantly faster than RNNs because convolution operations are fully
parallelizable on modern hardware.

**Two CNN architectures are trained and compared:**
1. **Simple CNN** — stacked Conv1D layers with different kernel sizes
2. **Multi-Scale CNN (TextCNN)** — parallel branches with different kernel sizes,
   concatenated before the classification head

Both models use **pretrained GloVe-100 embeddings** via Gensim.

---

## Files

| File | Description |
|------|-------------|
| `model3_cnn_1d.ipynb` | Training notebook |
| `results_model3_cnn.json` | Accuracy, AUC, classification report |
| `model3_cnn_simple.keras` | Saved Simple CNN model |
| `model3_cnn_multiscale.keras` | Saved Multi-Scale CNN model |
| `cnn_training_curves.png` | Validation curves comparison |
| `cnn_simple_cm.png` | Simple CNN confusion matrix |
| `cnn_multiscale_cm.png` | Multi-Scale CNN confusion matrix |

---

## Model Architectures

### Simple Stacked CNN

```
Input (integer tokens, length=256)
         │
         ▼
Embedding(20001 × 100)  ← GloVe-100 pretrained
         │
         ▼
SpatialDropout1D(0.3)
         │
         ▼
Conv1D(128 filters, kernel=3) → ReLU → MaxPool(2)
         │
         ▼
Conv1D(128 filters, kernel=4) → ReLU → MaxPool(2)
         │
         ▼
Conv1D(128 filters, kernel=5) → ReLU → MaxPool(2)
         │
         ▼
GlobalMaxPooling1D          ← fixed-size 128-dim vector
         │
         ▼
Dense(64) → ReLU → Dropout(0.3)
         │
         ▼
Dense(1) → Sigmoid
```

### Multi-Scale CNN (TextCNN — Kim 2014)

```
Input (integer tokens, length=256)
         │
         ▼
Embedding(20001 × 100)  ← GloVe-100 pretrained (shared)
         │
   ┌─────┼─────┐
   │     │     │
Conv(k=2) Conv(k=3) Conv(k=5)    ← parallel branches
   │     │     │
 Pool   Pool  Pool               ← GlobalMaxPooling each
   │     │     │
   └─────┼─────┘
         │ Concatenate [384-dim]
         │
         ▼
Dense(128) → ReLU → Dropout(0.3)
         │
         ▼
Dense(64)  → ReLU → Dropout(0.3)
         │
         ▼
Dense(1)   → Sigmoid
```

---

## How Convolution Works on Text

### 1. Token Embedding
Each token index is looked up in the embedding matrix → shape `(256, 100)`.

### 2. Conv1D Filter Sliding
A filter of shape `(kernel_size, 100)` slides along the sequence dimension.
- `kernel_size=2`: captures bigram patterns (2 consecutive words)
- `kernel_size=3`: captures trigram patterns
- `kernel_size=5`: captures 5-gram patterns

At each position, the filter computes a dot product with the local window of
word embeddings. After ReLU, negative values → 0; the output encodes
"how much does this window match this pattern?"

### 3. MaxPooling
`MaxPooling1D(pool_size=2)` takes the maximum over every 2 adjacent positions,
reducing sequence length by half. This provides translation invariance (the
pattern can appear anywhere in that window and still be detected).

### 4. GlobalMaxPooling1D
Takes the maximum across the **entire remaining sequence** per filter.
Result: one scalar per filter → 128-dim vector that captures the most
prominent occurrence of each learned pattern anywhere in the review.

### 5. Why Multiple Kernel Sizes?
Different kernel sizes capture patterns at different scales:
- Short kernels (2–3): catch local negations ("not good", "very bad")
- Longer kernels (5): catch phrasal patterns ("one of the best films")

The Multi-Scale CNN uses parallel branches so all scales contribute simultaneously.

---

## Techniques Applied

### Core Architectural Techniques
1. **Conv1D** — local pattern detection over token sequences
2. **Multiple kernel sizes** — multi-scale feature extraction
3. **GlobalMaxPooling1D** — position-invariant feature aggregation
4. **Parallel branches (TextCNN)** — simultaneous multi-scale processing

### Training Techniques
5. **Pretrained GloVe embeddings** — better word initialization than random
6. **Embedding fine-tuning** (`trainable=True`) — adapts pretrained vectors to IMDB
7. **SpatialDropout1D** — regularization on embedding feature maps
8. **Batch Normalization** (stacked CNN) — stable, faster training
9. **Adam optimizer** — adaptive learning rates
10. **Early Stopping** — halts training when validation loss plateaus
11. **ReduceLROnPlateau** — reduces lr when progress stalls

---

## CNN vs RNN for Text

| Aspect | 1D CNN | LSTM/GRU |
|--------|--------|----------|
| Training speed | Fast (parallelizable) | Slow (sequential) |
| Long-range dependencies | Limited | Good |
| Local pattern detection | Excellent | Moderate |
| Interpretability | Moderate | Low |
| Typical accuracy on IMDB | ~88–91% | ~88–93% |

CNNs are often preferred for sentiment tasks because:
- Sentiment is largely determined by local phrases ("terrible", "masterpiece")
- Much faster to train than RNNs
- Competitive accuracy despite simpler structure

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary size | 20,000 |
| Sequence length | 256 tokens |
| Embedding dimension | 100 (GloVe-100) |
| Number of filters | 128 |
| Kernel sizes (simple) | 3, 4, 5 (stacked) |
| Kernel sizes (multi-scale) | 2, 3, 5 (parallel) |
| Dropout rate | 0.3 |
| Batch size | 128 |
| Max epochs | 20 (early stopping) |
| Optimizer | Adam (lr=0.001) |

---

## How to Run

1. Open `model3_cnn_1d.ipynb` in Google Colab
2. Run all cells top-to-bottom
3. Simple CNN trains in ~5 min; Multi-Scale CNN ~8 min (CPU)

**GPU recommended for faster training:**
`Runtime → Change runtime type → T4 GPU`

### Requirements
```
tensorflow>=2.13
gensim>=4.3
nltk
kagglehub
scikit-learn>=1.3
```
