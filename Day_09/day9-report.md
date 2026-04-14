
# 📘 Day 9 Report: Hyperparameter Tuning (GridSearchCV)

## 🎯 Objective
The goal of Day 9 was to move from a standard machine learning model to an optimized model by applying hyperparameter tuning using GridSearchCV. The focus was on improving Ridge Regression performance on the California Housing dataset.

---

## 📊 Dataset Used
- California Housing Dataset (from sklearn)
- Features: Housing attributes ('MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude', 'MedHouseVal')
- Target: MedHouseVal

---

## ⚙️ Methodology

### 1. Baseline Model (Day 8)
A basic Ridge Regression model was trained using default parameters.

- Model: Ridge Regression
- No hyperparameter tuning
- Evaluation metric: R² Score

**Result:**
- Default R² Score: 0.67 (approx.)

---

### 2. Hyperparameter Tuning (Day 9)

GridSearchCV was applied to find the best value of alpha.

#### Parameter Grid:
```python
alpha = [0.1, 1, 10, 100, 500]
````

#### Technique Used:

* GridSearchCV
* Cross-validation (cv = 5)
* Scoring metric: R²

---

## Model Output

* Best Alpha Found: **10.0**
* Best Cross-Validation R² Score: **0.6991**

---

## 📈 Model Comparison

| Model       | Description                  | R² Score |
| ----------- | ---------------------------- | -------- |
| Day 8 Model | Default Ridge Regression     | 0.67     |
| Day 9 Model | GridSearchCV Optimized Ridge | 0.6991   |

---

## 🧠 Key Learnings

* Hyperparameters control model behavior and must be tuned for better performance.
* GridSearchCV automates hyperparameter selection and improves efficiency.
* Cross-validation ensures the model is evaluated fairly and avoids “luck-based” results.
* Ridge Regression helps reduce overfitting using regularization.
* Polynomial Regression helps model non-linear relationships in data.

---

## ✍️ Reflection

It is better to use a wider range of values (such as [0.1, 1, 10, 100]) first because we do not know the optimal region initially. A broad search helps identify the correct scale quickly. After finding the best range, smaller values can be used for fine-tuning. This approach saves computation time and improves efficiency.