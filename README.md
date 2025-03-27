# Spam Detection Using Machine Learning

This project focuses on building a spam detection system using Natural Language Processing (NLP) and machine learning. It classifies SMS messages as either **spam** or **ham** (not spam) using a supervised learning approach. The project is part of the **DLBAIPNLP01 – NLP Course** in the **Bachelor of Science in Data Science** program.

---

## Overview

With the growing use of digital communication, spam messages have become increasingly common and disruptive. This project demonstrates how machine learning and text classification techniques can be applied to detect and filter out spam messages in SMS communication.  

The primary model used is **Multinomial Naïve Bayes**, chosen for its effectiveness in text classification tasks. The project includes steps such as preprocessing, vectorization using **TF-IDF**, model training, evaluation, and result analysis.

---

## Project Structure

- Data cleaning and preprocessing using regular expressions
- Text vectorization using **TF-IDF**
- Model training with **Naïve Bayes**
- Evaluation using **accuracy, precision, recall, and F1-score**
- Confusion matrix generation
- Clean and reproducible pipeline

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tatjanakiriakov/spam-detection-nlp.git
   cd spam-detection-nlp

  python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

python src/preprocess.py
python train_model.py
python evaluate_model.py
python predict.py --message "Congratulations! You've won a free ticket"


# Model Performance
# Metric	Score
# Accuracy	98.5%
# Precision	97.2%
# Recall	96.8%
# F1-Score	97.0%

