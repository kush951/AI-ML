# 🚀 Day 4 – AI/ML Pre-processing Protocol

## 📌 Overview
Day 4 focused on **data preprocessing**, which is one of the most important steps in any machine learning pipeline.  
In real-world projects, raw data is often incomplete, inconsistent, and unscaled. This day was dedicated to transforming raw data into **clean, structured, and model-ready data**.

---

## 🎯 Objectives
- Handle missing values in datasets  
- Apply data imputation techniques  
- Perform feature scaling (normalization)  
- Understand relationships using data visualization  
- Build a complete preprocessing pipeline  

---

## 🔧 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🔄 Workflow

### 1️⃣ Data Cleaning
- Identified missing values using `isnull()`
- Filled missing values using:
  - Mean (for Age)
  - Median / Default (for Salary/Score)
- Rounded Age to maintain realistic values  
- Removed duplicates  

---

### 2️⃣ Feature Engineering (Scaling)
- Applied **MinMaxScaler** from Scikit-learn  
- Transformed features into range **[0,1]**  
- Ensured fair contribution of all features  

---

### 3️⃣ Exploratory Data Analysis (EDA)

Performed multiple visualization techniques:

#### 📊 Heatmap
- Showed strong positive correlations  
- Experience highly correlated with Salary  

#### 🔗 Pairplot
- Clear linear relationships between features  
- Experience strongly influences Salary  

#### 📦 Boxplot
- No major outliers detected  
- Salary shows highest variation  

#### 📉 Histogram
- Data is evenly distributed  
- No major skewness observed  

---

### 4️⃣ Experiment (Jupyter Notebook)
- Compared:
  - Dropping missing values  
  - Filling missing values  
- Concluded that **imputation is better for small datasets**

---

## 🧠 Key Learnings

- 80% of machine learning work is **data preparation**  
- Missing values must be handled carefully  
- Scaling improves model performance  
- Visualization helps in understanding data patterns  
- Clean data is essential for building accurate models  

---

## 🐞 Bug Encountered

**Error:**  
OSError: Cannot save file into a non-existent directory  

**Fix:**  
Used:
```python
import os
os.makedirs("outputs", exist_ok=True)