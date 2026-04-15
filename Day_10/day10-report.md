# 📄 Day 10: Non-Linear Models & Overfitting

## 🎯 Objective
The objective of this experiment is to understand how machine learning models capture non-linear relationships and how model complexity can lead to overfitting. We compare Polynomial Regression and Decision Tree Regressor and analyze the effect of max_depth.

---

## 🔹 1. Polynomial Regression (Curvy Model)

Polynomial Regression extends linear models by adding higher-degree features such as x² and x³, allowing the model to capture curved relationships.

### ✅ Observations:
- Produces a smooth curve  
- Captures non-linear patterns effectively  
- Degree 2 → Underfitting  
- Degree 3 → Good fit  
- Higher degree → Risk of overfitting ⚠️  

---

## 🔹 2. Decision Tree Regressor (Step Model)

Decision Trees split the data into regions using decision rules and assign constant values to each region.

### ✅ Observations:
- Produces step-like predictions  
- Captures non-linear relationships  
- Easy to interpret (Explainable AI)  
- Not smooth like polynomial models  

---

## 🔹 3. Effect of max_depth

The max_depth parameter controls the complexity of the Decision Tree.

### 📊 Observations:
- max_depth = 2 → Underfitting ❌  
- max_depth = 5 → Balanced model ✅  
- max_depth = 20 → Overfitting ⚠️  

---

## 🔹 4. Overfitting Analysis

Overfitting occurs when a model memorizes the training data instead of learning the underlying pattern.

### 📌 Observations:
- Model fits training data almost perfectly  
- Prediction curve becomes jittery and irregular  
- Poor performance on unseen data  

---

## 🔹 5. Model Comparison (R² Score)

- Polynomial Regression R²: 0.8525  
- Decision Tree R²: 0.8761  

### 📊 Insight:
- Decision Tree has higher R² → better training fit  
- Polynomial model provides smoother predictions  
- Higher R² does not always mean a better model  

---

## 🔹 6. Explainable AI (Decision Tree)

Decision Trees are considered Explainable AI because they can be visualized as flowcharts. Each decision is based on clear conditions, making the model transparent and interpretable.

---

## 🧠 Final Conclusion

Polynomial Regression provides a smooth approximation of the data and generalizes well. Decision Trees offer higher flexibility and better training performance but can easily overfit if not controlled.

Thus, selecting the right model and controlling complexity is crucial for achieving a balance between accuracy and generalization.

---

## 💡 Key Learnings

- Non-linear problems require advanced modeling techniques  
- Model complexity must be controlled  
- Overfitting reduces real-world performance  
- Decision Trees are powerful but need tuning  
- Smooth models often generalize better than complex ones  

---

