"""
intent_classifier_dl/train.py
-----------------------------
Training, evaluation, and comparison script for the Deep Learning Intent Classifier.
Fits ML Baselines and fine-tunes a Hugging Face DistilBERT model.
Compares models across performance metrics and confusion matrices.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from intent_classifier_dl.dataset import DATASET

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Class labels list
CLASSES = ["coding", "math", "ml", "deep_learning", "genai", "research", "system_design"]
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Custom PyTorch Dataset mapping Hugging Face tokens
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def main():
    logger.info("Initializing Deep Learning Intent Classifier Pipeline...")

    # Flatten and prepare labeled training dataset
    queries = []
    labels = []
    for expert, examples in DATASET.items():
        for example in examples:
            queries.append(example)
            labels.append(CLASS_TO_ID[expert])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        queries, labels, test_size=0.2, random_state=42, stratify=labels
    )

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    logger.info("Splits configured: Train count = %d, Test count = %d", len(X_train_text), len(X_test_text))

    # Store metrics dictionary for comparison
    model_metrics = {}
    confusion_matrices = {}

    # ---------------------------------------------------------------------------
    # 1. Train and Evaluate Baseline ML Models (TF-IDF Vectorization)
    # ---------------------------------------------------------------------------
    logger.info("Fitting and evaluating baseline TF-IDF classifiers...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vectorized = vectorizer.fit_transform(X_train_text)
    X_test_vectorized = vectorizer.transform(X_test_text)

    ml_baselines = {
        "Logistic Regression": LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Linear)": SVC(C=1.0, kernel="linear", probability=True, random_state=42)
    }

    for name, model in ml_baselines.items():
        model.fit(X_train_vectorized, y_train)
        preds = model.predict(X_test_vectorized)
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=range(len(CLASSES)))
        
        model_metrics[name] = {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
        confusion_matrices[name] = cm
        logger.info("- Baseline '%s': F1 Score = %.2f%%", name, f1 * 100)

    # ---------------------------------------------------------------------------
    # 2. Configure DistilBERT fine-tuning pipeline
    # ---------------------------------------------------------------------------
    logger.info("Loading pre-trained DistilBERT tokenizer and model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Instantiate classification model matching the 7 expert output classes
    dl_model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASSES),
        id2label=ID_TO_CLASS,
        label2id=CLASS_TO_ID
    )

    # Tokenize input texts
    logger.info("Tokenizing datasets for DistilBERT...")
    train_encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=64)

    # Wrap in PyTorch datasets
    train_dataset = IntentDataset(train_encodings, y_train)
    test_dataset = IntentDataset(test_encodings, y_test)

    # Metric evaluation callable
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    # Training arguments: Epoch is restricted to 1 as requested to avoid overfitting
    # on small dataset, but fully compiles the training pipeline structures
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1.0,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=dl_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    logger.info("Training/Inference of DistilBERT sequence classifier...")
    trainer.train()

    # Get predictions
    eval_results = trainer.predict(test_dataset)
    logits = eval_results.predictions
    dl_preds = np.argmax(logits, axis=-1)

    # Calculate DistilBERT metrics
    dl_acc = accuracy_score(y_test, dl_preds)
    dl_precision, dl_recall, dl_f1, _ = precision_recall_fscore_support(y_test, dl_preds, average="weighted", zero_division=0)
    dl_cm = confusion_matrix(y_test, dl_preds, labels=range(len(CLASSES)))

    model_metrics["DistilBERT"] = {
        "Accuracy": dl_acc,
        "Precision": dl_precision,
        "Recall": dl_recall,
        "F1 Score": dl_f1
    }
    confusion_matrices["DistilBERT"] = dl_cm
    logger.info("- Model 'DistilBERT': F1 Score = %.2f%%", dl_f1 * 100)

    # ---------------------------------------------------------------------------
    # 3. Save Deep Learning Model Components
    # ---------------------------------------------------------------------------
    dl_save_path = Path("intent_classifier_dl/model")
    dl_save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving DistilBERT model and tokenizer components to: %s", dl_save_path)
    dl_model.save_pretrained(dl_save_path)
    tokenizer.save_pretrained(dl_save_path)

    # ---------------------------------------------------------------------------
    # 4. Generate the Comparison Evaluation Report
    # ---------------------------------------------------------------------------
    report_path = Path("intent_classifier_dl/evaluation_report.md")
    logger.info("Writing evaluation comparison report to: %s", report_path)
    
    # Determine best model
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]["F1 Score"])
    best_model_score = model_metrics[best_model_name]["F1 Score"]

    # Build metric markdown table
    metrics_table = "| Classifier Model | Accuracy | Precision | Recall | F1 Score |\n| :--- | :--- | :--- | :--- | :--- |\n"
    for m_name, scores in model_metrics.items():
        metrics_table += f"| **{m_name}** | {scores['Accuracy']*100:.2f}% | {scores['Precision']*100:.2f}% | {scores['Recall']*100:.2f}% | {scores['F1 Score']*100:.2f}% |\n"

    # Build confusion matrices presentation
    cm_sections = ""
    for m_name, cm in confusion_matrices.items():
        cm_sections += f"\n### 📊 Confusion Matrix: {m_name}\n"
        cm_sections += "| True \\ Predicted | " + " | ".join([c.replace("_", " ").title() for c in CLASSES]) + " |\n"
        cm_sections += "| :--- | " + " | ".join([":---:" for _ in range(len(CLASSES))]) + " |\n"
        for idx, row in enumerate(cm):
            row_vals = " | ".join(str(val) for val in row)
            cm_sections += f"| **{CLASSES[idx].replace('_', ' ').title()}** | {row_vals} |\n"

    report_content = f"""# Evaluation Report: Deep Learning Intent Classifier

This report evaluates and compares traditional machine learning classifiers against the Hugging Face **DistilBERT** deep learning transformer model for routing queries in IntelliMoE.

---

## 📈 Performance Comparison Matrix

{metrics_table}

---

## 🏆 Model Verdict

Based on the macro-weighted F1 Score:
- **Best Performing Model**: **{best_model_name}** (F1 Score: **{best_model_score*100:.2f}%**)

---

## 🧬 Confusion Matrices

{cm_sections}

---

## ⚙️ Hyperparameters & Training Run Settings

- **DistilBERT Base Model**: `distilbert-base-uncased` (Sequence classification architecture with 7 output nodes)
- **Token Max Sequence Length**: 64 tokens
- **Training Epochs**: 1.0 (restricted to establish pipeline and prevent overfitting on constrained sample sizes)
- **Optimizer**: AdamW (Learning rate: `5e-5`, Warmup steps: `5`, Weight decay: `0.01`)
- **Dataset Size**: 210 queries (80% Train split, 20% Stratified Validation Test split)
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Evaluation report successfully written. Pipeline complete! 🎉")


if __name__ == "__main__":
    main()
