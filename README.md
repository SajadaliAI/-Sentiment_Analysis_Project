# 😀 Sentiment Analysis using NLP & Machine Learning

This project builds a **Sentiment Analysis model** that classifies text data (e.g., reviews, tweets) as **Positive, Negative, or Neutral** using NLP techniques and machine learning.

---

## 🎯 Objective
To analyze textual data and automatically determine sentiment polarity, helping businesses understand customer opinions and feedback.

---

## 📊 Dataset Overview
Dataset: `reviews.csv`  

**Columns include:**  
- `review_text` → Text of the review  
- `sentiment` → Target variable (`Positive`, `Negative`, `Neutral`)  

---

## 🧩 Steps Performed

### 🧹 1. Data Preprocessing
- Handled missing values  
- Text cleaning: lowercasing, punctuation removal, stopwords removal  
- Tokenization, lemmatization  

### 📊 2. Exploratory Data Analysis (EDA)
- Distribution of sentiments  
- Common words in positive and negative reviews  
- Word clouds and frequency analysis  

### 🤖 3. Feature Extraction
- Bag-of-Words / TF-IDF vectorization  
- Optional: Word embeddings (Word2Vec / GloVe)  

### 📈 4. Model Training
- Logistic Regression  

### 📈 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC Curve (for binary classification)  

### 💾 6. Model Saving
import pickle

# Make sure 'model' folder exists
import os
if not os.path.exists("model"):
    os.makedirs("model")

# Save trained Logistic Regression model
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

