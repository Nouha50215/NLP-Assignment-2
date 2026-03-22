# Model 2 — LSTM & GRU with Pretrained Word2Vec Embeddings

## Overview

This model uses Recurrent Neural Networks (RNNs) — specifically LSTM and GRU —
which are purpose-built for sequential data like text. Unlike the MLP in
Assignment 1 that sees reviews as a fixed feature vector, RNNs process word tokens
one by one, maintaining a hidden state that accumulates context.

**Two models are trained and compared:**
1. **Bidirectional LSTM** — reads the sequence forward and backward
2. **Bidirectional GRU** — lightweight variant with comparable performance

Both use **pretrained GloVe word embeddings** (loaded via Gensim) as the input
representation, providing richer semantic word vectors than random initialization.

---

## Files

| File | Description |
|------|-------------|
| `model2_rnn_lstm_gru.ipynb` | Training notebook |
| `results_model2_rnn.json` | Accuracy, AUC, classification report for both models |
| `model2_lstm.keras` | Saved LSTM model |
| `model2_gru.keras` | Saved GRU model |
| `rnn_training_curves.png` | Validation loss & accuracy comparison |
| `lstm_confusion_matrix.png` | LSTM confusion matrix |
| `gru_confusion_matrix.png` | GRU confusion matrix |

---

## Model Architectures

### LSTM Architecture

```
Input (integer token sequence, length=256)
         │
         ▼
Embedding(20001 × 100)   ← initialized with GloVe-100 weights
         │
         ▼
SpatialDropout1D(0.3)    ← drops entire feature maps
         │
         ▼
Bidirectional LSTM(64)   ← forward + backward = 128 units, return_sequences=True
  [forget gate: ft = σ(Wf·[ht-1, xt] + bf)]
  [input gate:  it = σ(Wi·[ht-1, xt] + bi)]
  [output gate: ot = σ(Wo·[ht-1, xt] + bo)]
  [cell state:  Ct = ft⊙Ct-1 + it⊙tanh(Wc·[ht-1, xt])]
         │
         ▼
LSTM(32)                 ← second layer, return_sequences=False
         │
         ▼
Dense(32) → ReLU → Dropout(0.3)
         │
         ▼
Dense(1) → Sigmoid
```

### GRU Architecture

```
Input (integer token sequence, length=256)
         │
         ▼
Embedding(20001 × 100)   ← initialized with GloVe-100 weights
         │
         ▼
SpatialDropout1D(0.3)
         │
         ▼
Bidirectional GRU(64)    ← forward + backward = 128 units
  [update gate: zt = σ(Wz·[ht-1, xt])]
  [reset gate:  rt = σ(Wr·[ht-1, xt])]
  [candidate:   h̃t = tanh(W·[rt⊙ht-1, xt])]
  [hidden:      ht = (1-zt)⊙ht-1 + zt⊙h̃t]
         │
         ▼
GRU(32)
         │
         ▼
Dense(32) → ReLU → Dropout(0.3)
         │
         ▼
Dense(1) → Sigmoid
```

---

## Pretrained Word Embeddings

### What are Word Embeddings?
Each word is mapped to a dense vector (100 dimensions) that encodes semantic meaning.
Similar words have similar vectors (e.g., "great" and "excellent" are close in vector space).

### Why Pretrained?
Training word vectors from scratch requires enormous amounts of text.
Pretrained embeddings (GloVe/Word2Vec) are trained on billions of words and
capture rich linguistic knowledge that improves downstream tasks.

### How We Use GloVe
1. Load `glove-wiki-gigaword-100` via Gensim (trained on Wikipedia + Gigaword corpus)
2. Build an embedding matrix: row `i` = GloVe vector for word with index `i`
3. Initialize the Keras `Embedding` layer with this matrix
4. Set `trainable=True` to allow fine-tuning on IMDB data

### Alternative: Word2Vec (Google News)
To use the higher-quality Word2Vec-300 vectors, change:
```python
word_vectors = gensim_api.load('word2vec-google-news-300')
EMBEDDING_DIM = 300
```

---

## Key Concepts

### LSTM Gates
LSTMs use three gates to control information flow:
- **Forget gate** `ft`: decides what to erase from cell state
- **Input gate** `it`: decides what new information to store
- **Output gate** `ot`: decides what part of cell state to output

This gating mechanism allows LSTMs to capture dependencies over 100+ time steps,
solving the vanishing gradient problem of vanilla RNNs.

### GRU vs LSTM
| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Parameters | More | Fewer |
| Training speed | Slower | Faster |
| Performance | Often slightly better | Often comparable |

### Bidirectional RNNs
A standard RNN only sees past context. A Bidirectional RNN runs two RNNs:
- One left-to-right (past context)
- One right-to-left (future context)
Both hidden states are concatenated, giving the model full context.

### SpatialDropout1D
Standard Dropout randomly zeros individual activations.
SpatialDropout1D zeros entire feature maps (channels), which is more effective
for embeddings since adjacent time steps are highly correlated.

---

## Techniques Applied

1. **Pretrained GloVe embeddings** (Gensim) — semantic word initialization
2. **Fine-tuning** — embedding weights continue to update during training
3. **Bidirectional RNNs** — captures both past and future context
4. **Stacked RNN layers** — deeper representation extraction
5. **SpatialDropout1D** — regularization for embedding layers
6. **Recurrent Dropout** — dropout on hidden-to-hidden connections
7. **Adam optimizer** with adaptive learning rate
8. **Early Stopping** — prevents overfitting
9. **ReduceLROnPlateau** — fine-grained lr scheduling

---

## Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary size | 20,000 |
| Sequence length | 256 tokens |
| Embedding dimension | 100 (GloVe-100) |
| LSTM/GRU units (layer 1) | 64 (×2 bidirectional = 128) |
| LSTM/GRU units (layer 2) | 32 |
| Dropout rate | 0.3 |
| Recurrent dropout | 0.1 |
| Batch size | 128 |
| Max epochs | 15 (early stopping) |
| Optimizer | Adam (lr=0.001) |


### Requirements
```
tensorflow>=2.13
gensim>=4.3
nltk
kagglehub
scikit-learn>=1.3
```
