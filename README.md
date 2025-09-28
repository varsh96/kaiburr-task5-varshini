"# Kaiburr Task 5  mutyala varshini" 
# Consumer Complaint Text Classification

This project performs **text classification** on the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) from the U.S. Consumer Financial Protection Bureau (CFPB).  

The goal is to classify consumer complaints into **four categories**:


---

## 🚀 Steps Implemented

### 1. Explanatory Data Analysis (EDA) & Feature Engineering
- Loaded the dataset (`Consumer_Complaints.csv`).
- Selected the complaint narrative column as input text.
- Mapped the **Product** column into target categories (0–3).
- Added text length as a basic exploratory feature.
- Visualized class distribution with plots (optional).

### 2. Text Preprocessing
- Lowercased text, removed URLs, emails, and punctuation.  
- Removed stopwords and short tokens.  
- Applied **lemmatization** using NLTK.  
- Created a cleaned text column (`clean_text`).  
- Verified preprocessing with before/after samples.

### 3. Model Selection
- Defined multiple candidate models:
  - Logistic Regression
  - Naive Bayes
  - Linear SVM
  - Random Forest
- Used **TF-IDF vectorization** with unigrams & bigrams (`max_features=20000`).

### 4. Model Training & Comparison
- Split data into train/test sets (80/20, stratified).  
- Built pipelines for each model.  
- Evaluated models using:
  - Accuracy  
  - Precision / Recall / F1 (macro-averaged)  
  - 5-fold cross-validation  

### 5. Model Evaluation
- Selected the **best model** (based on macro F1).  
- Plotted a **confusion matrix heatmap**.  
- Printed a detailed **classification report**.  
- Displayed top predictive features (for linear models).  
- Saved the best model to `best_text_classifier.joblib`.

### 6. Prediction
- Printed the category mapping in the required format:


- Tested the classifier with new sample complaints:
  - Example: *"Debt collector keeps calling me for an old debt that I don't owe"* → Predicted as **1 Debt collection**

---

## Project Structure

     main.py 
     Consumer_Complaints.csv 
     best_text_classifier.joblib 
     requirements.txt 
     README.md 
---

##  Installation & Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt

pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
joblib










