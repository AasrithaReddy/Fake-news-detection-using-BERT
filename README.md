Dataset Description
train.csv: A full training dataset with the following attributes:

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
1: unreliable
0: reliable
test.csv: A testing training dataset with all the same attributes at train.csv without the label.

submit.csv: A sample submission that you can

# 📰 Fake News Detection using BERT

## 📌 Overview
This project is a Natural Language Processing (NLP) solution to detect fake news articles using **BERT (Bidirectional Encoder Representations from Transformers)**. By fine-tuning a pre-trained BERT model, we aim to classify news text as **REAL** or **FAKE** with high accuracy.

## 🚀 Features
- Cleaned and preprocessed real-world fake news datasets
- Tokenization and embedding using BERT tokenizer
- Fine-tuning with `bert-base-uncased` model
- Evaluation using Accuracy, Precision, Recall, and F1 Score
- Optional web app deployment using **Streamlit**

## 🗃 Dataset
We use publicly available datasets such as:
- [LIAR Dataset (UCSB)]
- [Kaggle Fake News Dataset]
Each dataset includes labeled statements or articles marked as "real" or "fake".

## 🧹 Preprocessing
- Lowercasing
- Removing stop words and special characters
- Tokenizing with `BertTokenizer` (from Hugging Face Transformers)
- Padding and truncating sequences to BERT-compatible lengths

## 🧠 Model Architecture
- Pre-trained BERT model: `bert-base-uncased`
- Classification head with 2 output labels
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss

## 🏋️‍♂️ Training
- Framework: PyTorch (can be adapted to TensorFlow)
- Batch size: 16
- Epochs: 3
- Evaluation strategy: Per epoch
- Fine-tuned using Hugging Face `Trainer` API

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## 🌐 Deployment
You can deploy the trained model using **Streamlit** or **Flask**:

### Streamlit Example:
```python
import streamlit as st
from transformers import pipeline

classifier = pipeline("text-classification", model="path_to_model")

st.title("Fake News Detector")
text = st.text_area("Enter news text:")
if st.button("Detect"):
    result = classifier(text)
    st.write("Prediction:", result[0]['label'])
```

## 📁 Project Structure
```
├── fake-news-detection-using-bert-pytorch.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── liar.csv or train.csv
├── model/
│   └── saved_model/
└── app/
    └── streamlit_app.py
```

## 📦 Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install transformers pandas scikit-learn streamlit torch
```

## 📚 Credits
Inspired by research from UCSB and Kaggle open data challenges. Built using Hugging Face, PyTorch, and Streamlit.


