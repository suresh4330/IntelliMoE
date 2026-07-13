# Evaluation Report: Deep Learning Intent Classifier

This report evaluates and compares traditional machine learning classifiers against the Hugging Face **DistilBERT** deep learning transformer model for routing queries in IntelliMoE.

---

## 📈 Performance Comparison Matrix

| Classifier Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **SVM (Linear)** | 95.24% | 95.83% | 95.24% | 95.18% |
| **Logistic Regression** | 92.86% | 93.65% | 92.86% | 92.74% |
| **Random Forest** | 90.48% | 91.27% | 90.48% | 90.35% |
| **DistilBERT (1 Epoch)** | 19.05% | 12.30% | 19.05% | 13.56% |

---

## 🏆 Model Verdict

Based on the macro-weighted F1 Score:
- **Best Performing Model**: **SVM (Linear)** (F1 Score: **95.18%**)
- **Verdict Explanation**: 
  Traditional ML models (SVM, Logistic Regression) perform exceptionally well because our 210-query dataset contains highly distinct class-specific keywords (e.g. *quicksort*, *integral*, *batch size*, *sharding*) which can be easily separated by linear classifiers in a sparse TF-IDF space. 
  DistilBERT, with its randomly initialized classification head, yields a low F1 Score after just 1 epoch because deep transformer networks have millions of parameters and require a significantly larger corpus and many more optimization epochs to train the classification boundaries effectively.

---

## 🧬 Confusion Matrices

### 📊 Confusion Matrix: SVM (Linear)
| True \ Predicted | Coding | Math | ML | Deep Learning | GenAI | Research | System Design |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coding** | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Math** | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| **ML** | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| **Deep Learning** | 0 | 0 | 1 | 5 | 0 | 0 | 0 |
| **GenAI** | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| **Research** | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| **System Design** | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

### 📊 Confusion Matrix: Logistic Regression
| True \ Predicted | Coding | Math | ML | Deep Learning | GenAI | Research | System Design |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coding** | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Math** | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| **ML** | 0 | 0 | 5 | 1 | 0 | 0 | 0 |
| **Deep Learning** | 0 | 0 | 2 | 4 | 0 | 0 | 0 |
| **GenAI** | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| **Research** | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| **System Design** | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

### 📊 Confusion Matrix: Random Forest
| True \ Predicted | Coding | Math | ML | Deep Learning | GenAI | Research | System Design |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coding** | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Math** | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| **ML** | 0 | 0 | 4 | 2 | 0 | 0 | 0 |
| **Deep Learning** | 0 | 0 | 2 | 4 | 0 | 0 | 0 |
| **GenAI** | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| **Research** | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| **System Design** | 0 | 0 | 0 | 0 | 0 | 0 | 6 |

### 📊 Confusion Matrix: DistilBERT (1 Epoch)
| True \ Predicted | Coding | Math | ML | Deep Learning | GenAI | Research | System Design |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Coding** | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| **Math** | 2 | 1 | 0 | 1 | 0 | 1 | 1 |
| **ML** | 1 | 1 | 0 | 2 | 1 | 0 | 1 |
| **Deep Learning** | 0 | 1 | 1 | 2 | 1 | 1 | 0 |
| **GenAI** | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| **Research** | 2 | 1 | 0 | 1 | 1 | 1 | 0 |
| **System Design** | 1 | 1 | 1 | 1 | 1 | 0 | 1 |

---

## ⚙️ Hyperparameters & Training Run Settings

- **DistilBERT Base Model**: `distilbert-base-uncased` (Sequence classification architecture with 7 output nodes)
- **Token Max Sequence Length**: 64 tokens
- **Training Epochs**: 1.0 (restricted to establish pipeline and prevent overfitting on constrained sample sizes)
- **Optimizer**: AdamW (Learning rate: `5e-5`, Warmup steps: `5`, Weight decay: `0.01`)
- **Dataset Size**: 210 queries (80% Train split, 20% Stratified Validation Test split)
