# 🚀 Day 5 – Linear Regression & Evaluation (AI/ML Track)

## 🔧 Technical Summary
Today I implemented my first machine learning model using **Linear Regression** and learned how to evaluate its performance.

I worked on:
- Splitting data into training and testing sets (Train-Test Split)
- Training a Linear Regression model using `fit()`
- Making predictions using `predict()`
- Evaluating model performance using **MSE** and **R² Score**
- Visualizing the regression line using Matplotlib
- Understanding the impact of training data size on model performance

This helped me understand how a model learns patterns from data and how we measure its accuracy.

---

## 📊 Model Performance
### Notebook file (linear_Regression.ipynb)
Consider the data set (Salary_datasets.csv)
- **Mean Squared Error (MSE):** 49830096.86  
- **R-Squared Score (R²):** 0.90  
---
### Script File (Task Sample dataset )
- **Mean Squared Error:** 17.89 
- **R-Squared Score:** 0.97
---

## 📈 Observations

- The model achieved a high **R² score of 0.90**, indicating strong performance  
- It explains **90% of the variance** in salary based on experience  
- The regression line fits closely to most data points  
- The **MSE appears high** due to the large scale of salary values  
- Predictions are generally close to actual values  
- More training data improves model accuracy and stability  

---

## 🧪 Experiment Insight

I compared:
- Model trained on full dataset  
- Model trained on only 2 data points  

### Result:
- The full model captured the overall trend accurately  
- The 2-point model failed to generalize  
- Less data leads to poor predictions and overfitting  

---

## 🧠 Key Learning

- Models should always be tested on unseen data  
- More data leads to better generalization  
- Evaluation metrics like MSE and R² are essential to judge performance  
- Visualization helps in understanding model behavior  

---

## 🐞 Bug Log

**Issue Faced:**  
Model prediction error due to incorrect input shape  

**Root Cause:**  
Scikit-learn expects input in 2D format, but a 1D value was passed  

**Fix Applied:**  
Converted input into 2D format:

```python
model.predict([[value]])