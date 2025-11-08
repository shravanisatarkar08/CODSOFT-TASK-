# CODSOFT-TASK-
# Task 1 - Credit Card Fraud Detection

## Overview
Developed a machine learning model to detect fraudulent credit card transactions using numerical transaction data.

## Dataset
- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with 492 fraudulent cases.

## Technologies Used
- Python, Pandas, NumPy, Scikit-learn
- Algorithms: Random Forest Classifier
- Feature Scaling: StandardScaler

## Steps Performed
1. Data Loading and Exploration  
2. Train-Test Split  
3. Data Standardization  
4. Model Training using RandomForestClassifier  
5. Model Evaluation using Accuracy and Confusion Matrix  

## Results
- Achieved ~99.9% Accuracy on Test Data  
- Detected fraudulent transactions effectively
- Accuracy: 0.9991
Confusion Matrix:
[[56861 5]
[ 52 44]]

# Task 2 - Spam SMS Detection using TF-IDF + Naive Bayes

## 📌 Overview
This project builds a **machine learning model** to classify SMS messages as **Spam** or **Not Spam (Ham)**.  
It uses **TF-IDF vectorization** to convert text messages into numerical features and applies the **Multinomial Naive Bayes** algorithm for classification.

---

## 🧾 Dataset
- **Source:** [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- The dataset contains:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: The message text

Example:

| v1   | v2                              |
|------|----------------------------------|
| ham  | I'm going to be home soon.      |
| spam | Congratulations! You won a prize! |

---

## ⚙️ Model Workflow

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Text feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
📈 Results
Model: Multinomial Naive Bayes

Feature Extraction: TF-IDF Vectorizer

Test Accuracy: ~97–99% (depending on data split)

📚 Dependencies
Make sure you have the following Python libraries installed:

bash
Copy code
pip install pandas scikit-learn


## Author
Shravani Satarkar  
CodSoft Machine Learning Internship (November 2025)
