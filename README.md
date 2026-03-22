# Assignment 2 — Sentiment Analysis: Model Improvement

Dataset: [IMDB Movie Reviews (50K)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
Task: Binary sentiment classification (positive / negative)
Baseline (Assignment 1): MLP with 5 VADER + TextBlob features → **77.42% accuracy**

---

## Repository Structure

├── model1_optimized_mlp.ipynb      Model 1 notebook
├── README_model1_mlp.md            Model 1 README
├── results_model1_mlp.json         Model 1 results
│
├── model2_rnn_lstm_gru.ipynb       Model 2 notebook
├── README_model2_rnn.md            Model 2 README
├── results_model2_rnn.json         Model 2 results (LSTM + GRU)
│
├── model3_cnn_1d.ipynb             Model 3 notebook
├── README_model3_cnn.md            Model 3 README
├── results_model3_cnn.json         Model 3 results (CNN variants)
│
└── README.md                      

---

## Models Summary

### Model 1 — Optimized MLP
**File:** `model1_optimized_mlp.ipynb`

An improved Multi-Layer Perceptron that extends Assignment 1 with:
- Richer features: TF-IDF (5000 dims) + VADER + TextBlob (5 dims) = 5005-dim input
- Batch Normalization + Dropout regularization
- Comparison of Adam vs SGD (with Nesterov momentum) vs RMSProp
- Early stopping & ReduceLROnPlateau callbacks
- StandardScaler feature normalization

Architecture: `Dense(256) → BN → ReLU → Dropout → Dense(128) → ... → Dense(1)`

---

### Model 2 — LSTM & GRU (RNNs)
**File:** `model2_rnn_lstm_gru.ipynb`

Recurrent Neural Networks that process reviews as token sequences.
Both architectures use **pretrained GloVe-100 word embeddings** loaded via Gensim.

**BiLSTM:**
`Embedding → SpatialDropout → BiLSTM(64) → LSTM(32) → Dense → Sigmoid`

**BiGRU:**
`Embedding → SpatialDropout → BiGRU(64) → GRU(32) → Dense → Sigmoid`

Key techniques: bidirectional processing, pretrained embeddings, recurrent dropout.

---

### Model 3 — 1D CNN (TextCNN)
**File:** `model3_cnn_1d.ipynb`

Convolutional networks that detect local n-gram patterns in the token sequence.
Two variants compared:

**Simple CNN:** stacked Conv1D layers (kernel sizes 3→4→5) + GlobalMaxPool

**Multi-Scale CNN (TextCNN):** parallel branches with kernel sizes 2, 3, 5
concatenated before the classification head — captures patterns at multiple scales simultaneously.

Both use **pretrained GloVe-100 embeddings** via Gensim with fine-tuning.

---

## Techniques Applied (per assignment requirements)

| Requirement | Where Applied |
|-------------|---------------|
| Mini-batch gradient descent | All models (batch_size=128/256) |
| SGD with Momentum / EWMA | Model 1 (optimizer comparison) |
| Adam (adaptive lr) | All models |
| RMSProp | Model 1 (optimizer comparison) |
| Batch Normalization | Model 1 |
| Hyperparameter tuning | Model 1 (layers, dropout, lr, optimizer) |
| RNN — LSTM | Model 2 |
| RNN — GRU | Model 2 |
| 1D CNN | Model 3 |
| Pretrained Word2Vec/GloVe embeddings | Models 2 & 3 (Gensim) |

---

## Results Comparison

| Model | Architecture | Key Feature | Test Accuracy | AUC |
|-------|-------------|-------------|---------------|-----|
| Baseline (Assign. 1) | MLP (64,32) | VADER + TextBlob (5 features) | 77.42% | — |
| Model 1 | MLP (256,128,64) + BN + Dropout | TF-IDF + Sentiment | see JSON | see JSON |
| Model 2a | Bidirectional LSTM | GloVe-100 embeddings | see JSON | see JSON |
| Model 2b | Bidirectional GRU | GloVe-100 embeddings | see JSON | see JSON |
| Model 3a | Simple 1D CNN | GloVe-100 embeddings | see JSON | see JSON |
| Model 3b | Multi-Scale CNN (TextCNN) | GloVe-100 embeddings | see JSON | see JSON |

*Run the notebooks to populate the results JSON files.*

---

## How to Run All Models

All notebooks are designed for **Google Colab**. For Models 2 and 3, enabling
a GPU runtime will significantly reduce training time.

### Steps:
1. Open notebook in Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU` (recommended for Models 2 & 3)
3. Run all cells top to bottom (`Runtime → Run all`)
4. Results are saved as `.json` files and plots as `.png` files

### Global Requirements
```
tensorflow>=2.13
scikit-learn>=1.3
gensim>=4.3
nltk
textblob
vaderSentiment
kagglehub
scipy
matplotlib
seaborn
```

---

## Assignment Reference

**Course:** [Your course name]
**Assignment 2 objectives:**
- Apply optimization techniques (SGD, Momentum, Adam, RMSProp, BN)
- Hyperparameter tuning
- Train LSTM / GRU variants
- Train 1D CNNs
- Use pretrained word embeddings (Gensim / Word2Vec / GloVe)
