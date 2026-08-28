# Tweet Sentiment Classification with DistilBERT

Fine-tune a [DistilBERT](https://arxiv.org/abs/1910.01108) model to classify the emotional state of tweets, and compare Transformer-based classification with classical scikit-learn baselines on fixed features.

## Overview

Course project for *Introduction to Artificial Intelligence* (Fudan University, Fall 2024). The goal: build a system that automatically recognizes the emotion expressed in a tweet — `sadness, joy, love, anger, fear, surprise` — using DistilBERT, a lightweight BERT variant.

The notebook is based on course-provided materials (following Ch. 2 of *Natural Language Processing with Transformers* by Tunstall, von Werra & Wolf, 2022) and completes the assignment tasks below.

### Assignment tasks completed

| Task | Implementation |
| --- | --- |
| 1. Character-level tokenization | `token2idx` dictionary mapping characters to integer ids |
| 2. `[CLS]` feature extraction | `extract_hidden_states()` returning the 768-d `[CLS]` hidden vector |
| 3. scikit-learn baselines | `LinearRegression`, `LogisticRegression`, `DummyClassifier` trained on `[CLS]` features |
| 4. Hyperparameter tuning | Tuned `batch_size=100`, `learning_rate=1e-4`, `num_train_epochs=6`, `weight_decay=0.01` (saved in `training_args.bin`) |

## Results

Validation results from the notebook:

| Model | Validation accuracy |
| --- | --- |
| DummyClassifier (most frequent) | 35.2% |
| LogisticRegression on `[CLS]` features | 63.5% |
| Fine-tuned DistilBERT | **94.1% accuracy / 94.1% F1** |

Fine-tuning a pretrained Transformer clearly beats linear classifiers on frozen features, confirming that task-specific fine-tuning captures tweet semantics far better than a linear model over `[CLS]` embeddings.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook tweet_sentiment_classification.ipynb
```

The notebook downloads the `emotion` dataset and the DistilBERT checkpoint from Hugging Face (internet connection needed on first run). GPU recommended for fine-tuning.

## Project structure

```
Tweet-Sentiment-Classification/
├── tweet_sentiment_classification.ipynb   # main notebook (tasks 1-4)
├── training_args.bin                      # tuned hyperparameters (Task 4)
├── requirements.txt
└── .gitignore
```

## Disclaimer

Educational project. The baseline notebook and assignment were provided by the course instructor; the completed tasks are the author's own work. The `emotion` dataset is from E. Saravia et al., *CARER: Contextualized Affect Representations for Emotion Recognition* (EMNLP 2018).
