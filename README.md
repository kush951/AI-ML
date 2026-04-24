
# 🚀 90 Days Industry Immersion Program (AI/ML Track)

## 👨‍💻 Overview

This repository tracks my **15-day progress** in the AI/ML track, focused on building **industry-level skills** in data science, machine learning, and real-world problem solving.

---

# 📅 Progress Summary (Day 1 – Day 15)

---

## 🧠 Phase 1: Foundations (Day 1 – Day 4)

### 🔹 Covered:

* Python Basics (loops, functions, logic)
* Data Structures (lists, dictionaries, tuples)
* NumPy & Pandas
* Data Cleaning & Preprocessing

### 💡 Key Takeaways:

* Built strong programming foundation
* Learned efficient data handling
* Understood importance of clean data

---

## 📊 Phase 2: Core Machine Learning (Day 5 – Day 10)

### 🔹 Models Implemented:

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)

### 🔹 Concepts Learned:

* Train-Test Split
* Model Training vs Evaluation
* Overfitting & Underfitting

### 📈 Evaluation Techniques:

* Confusion Matrix
* Accuracy, Precision, Recall, F1-score

### 💡 Key Insight:

> Accuracy alone is not enough — understanding errors is critical.

---

## 🌲 Phase 3: Advanced Models & Comparison (Day 11 – Day 13)

### 🔹 Models Explored:

* Random Forest
* Model Comparison Techniques

### 🔹 Work Done:

* Compared multiple models
* Analyzed feature importance
* Improved prediction performance

### 💡 Key Insight:

> Ensemble models like Random Forest improve stability and accuracy.

---

## 🚀 Phase 4: Real-World Project (Day 13 -14)

### 🎯 Project: **Mux Intelligence – Event Networking Matcher**

### 🔹 Objective:

Predict compatibility between users for networking events.

### 🔹 Features Used:

* Domain similarity
* Career goals
* Experience level
* Communication style
* Location

### 🔹 What I Built:

* Pairwise feature engineering system
* ML-based compatibility prediction
* Model evaluation pipeline
* Visualization (heatmaps, charts)

### 💡 Key Insight:

> Real-world ML is about **feature engineering + problem understanding**, not just models.

---

# 🛠️ Tech Stack

* Python
* NumPy & Pandas
* Scikit-learn
* Matplotlib & Seaborn
* Flask (API Integration)

---

## Day 15: Clustering & Visualization

🔹 Work Done:
* Implemented K-Means Clustering
* Built 2D & 3D cluster visualizations
* Analyzed user grouping patterns
* Generated visual outputs (charts, cluster plots)

🔹 Key Concepts:
* Unsupervised Learning
* Cluster formation
* Pattern discovery in data
💡 Key Insight:

**Key Insight:**

Clustering helps discover hidden patterns without labels — useful for segmentation and recommendations.

---

#  Day 16: Model Validation, K-Fold & Statistical Proof (AI/ML Track)

##  Overview

Day 16 focuses on **model reliability and statistical validation**. Instead of trusting a single train-test split, we evaluate models using **K-Fold Cross Validation, Stratified K-Fold, and statistical metrics like Mean Accuracy and Standard Deviation**.

This ensures that our model is **stable, unbiased, and generalizes well to real-world data**.

---

##  Objectives

- Implement K-Fold Cross Validation using `cross_val_score`
- Measure model performance across multiple folds
- Calculate **Mean Accuracy** and **Standard Deviation**
- Detect **Overfitting vs Underfitting**
- Understand the importance of **Shuffling in K-Fold**
- Apply **Stratified K-Fold for imbalanced datasets**
- Perform **consistency testing (shuffle vs no shuffle)**

---

##  Key Concepts Covered

###  K-Fold Cross Validation
Splits dataset into K parts and trains model K times to ensure robust evaluation.

---

###  Bias-Variance Tradeoff
- Overfitting → High train accuracy, low validation accuracy  
- Underfitting → Low train and validation accuracy  
- Good Model → Train ≈ Validation accuracy  

---

###  Stability Metrics

- Mean Accuracy → Overall performance  
- Standard Deviation → Stability across folds  

---

##  Example Results

### ✔ Cross Validation Output
- Fold Scores: `[0.93, 0.91, 0.96, 0.96, 0.94]`
- Mean Accuracy: **0.9399**
- Std Dev: **0.0206**

---

### ✔ Shuffle Test

| Case | Mean Accuracy | Std Dev |
|------|--------------|---------|
| Without Shuffle | 95.16% | 0.0243 |
| With Shuffle | 97.94% | 0.0108 |

Shuffling improves stability and reduces variance.

---

### ✔ Stratified K-Fold (Imbalanced Data)

- Mean Accuracy: **96.66%**
- Std Dev: **0.021**
- Max Accuracy: **100%**
- Min Accuracy: **92.98%**

---

##  Key Insights

- Cross-validation removes dependency on a single data split
- Shuffling ensures unbiased data distribution
- Stratified K-Fold preserves class balance
- Standard deviation is critical for measuring model stability
- A stable model is more important than a high single accuracy score

---

##  AI Pro Tip

> If Standard Deviation > 0.05 → Model is unstable and unreliable in production.

---

##  Real-World Application

In systems like **MeetMux**:
- User data may be imbalanced (e.g., Bangalore vs Delhi users)
- Without proper validation:
  - Model becomes biased toward majority class
- Solution:
  - Use Shuffle + Stratified K-Fold for fair evaluation

---

##  Conclusion

Day 16 teaches that:
- Model evaluation is not just about accuracy
- Stability and consistency are critical
- Proper validation techniques ensure real-world reliability
- A good ML model must be **accurate, stable, and generalizable**

---

#  Day 17: Hyperparameter Tuning

##  Objective
Optimize model performance using GridSearchCV and RandomizedSearchCV.

---

## ️ Model
- Random Forest Classifier  
- Dataset: Breast Cancer (sklearn)

---

##  Results
- **Best Parameters:**  
  {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}  

- **Best Accuracy:**  
  96.31%

---

##  Comparison

| Method              | Fits | Accuracy |
|--------------------|------|----------|
| GridSearchCV       | 81   | 96.31%   |
| RandomizedSearchCV | 30   | 96.31%   |

---

##  Key Insight
RandomizedSearchCV achieved the same accuracy with fewer computations (~63% faster).

---

##  Model Saved
- `best_model.pkl` using joblib

---

##  Takeaway
Efficient optimization > Exhaustive search

---