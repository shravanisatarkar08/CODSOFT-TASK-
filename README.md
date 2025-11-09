
> **Note:** All notebook code uses relative dataset paths (`datasets/fraud.csv`, etc.) so it runs locally or in Colab.

---

## Task 1: Fraud Detection  

**Objective:** Detect fraudulent transactions using Machine Learning.  

**Dataset:** `datasets/fraud.csv` (features + `Fraud` target column)  

**Approach:**  
- Load and clean dataset  
- Encode categorical variables if present  
- Split dataset into train and test sets  
- Train a classifier (Random Forest or Logistic Regression)  
- Evaluate using **accuracy**, **F1-score**, **confusion matrix**  

**Key Output:**  
- Accuracy: *Example: 94%*  
- Confusion Matrix: `[[TN FP], [FN TP]]`  

---

## Task 2: Spam Detection  

**Objective:** Classify SMS messages as **Spam** or **Ham**.  

**Dataset:** `datasets/spam.csv` (columns: `v1` = label, `v2` = message)  

**Approach:**  
- Load and clean dataset  
- Encode labels (`ham`=0, `spam`=1)  
- Split dataset into train/test sets  
- Convert text to **TF-IDF features**  
- Train **Multinomial Naive Bayes**  
- Evaluate using **accuracy**, **classification report**, and **confusion matrix**  
- Predict on sample messages  

**Key Output:**  
- Accuracy: 96.68%  
- Confusion Matrix: `[[965 0], [37 113]]`  
- Sample Prediction: `[1] → Spam`  

---

## Task 3: Churn Prediction  

**Objective:** Predict whether a customer will **churn** using customer data.  

**Dataset:** `datasets/churn.csv` (features include `customerID`, `gender`, `tenure`, target: `Churn`)  

**Approach:**  
- Load and clean dataset  
- Encode categorical variables  
- Split dataset into train/test sets  
- Scale numerical features  
- Train **Random Forest Classifier**  
- Evaluate using **accuracy**, **classification report**, and **confusion matrix**  

**Key Output:**  
- Accuracy: 86.65%  
- Confusion Matrix Example: `[[900 50], [100 65]]`  

---

## Notes  

- All tasks are implemented in **Python 3** in **Google Colab**.  
- Datasets are included in the `datasets/` folder.  
- Code uses **relative paths** for datasets to ensure smooth execution.  
- Notebooks contain **full code, outputs, and visualizations**.  

---

## Author  

**Shravani Satarkar**  
🌐 GitHub: [https://github.com/<shravanisatarkar08>](https://github.com/<your-username>)  
📧 Email: [shravani.satarkar08@gmail.com]  

---
ember 2025)
