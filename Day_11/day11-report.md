
#  Day 11 Report: Ensemble Learning & Model Evaluation

##  Objective
The objective of this experiment is to understand and apply Ensemble Learning techniques such as Bagging, Pasting, and Voting, and compare them with individual regression models like Linear Regression, Ridge Regression, and Random Forest. Also analyze the impact of hyperparameter tuning using GridSearchCV and RandomizedSearchCV.

---

#  Ensemble Learning Concepts

##  1. Bagging (Bootstrap Aggregating)
- Uses **bootstrap sampling (sampling with replacement)**
- Each model is trained on a different subset of data
- Helps reduce variance and overfitting
- On average, each model sees ~63.2% of data

---

##  2. Pasting
- `bootstrap = False`
- Sampling is done **without replacement**
- No repeated rows in training subsets

---

##  Bagging vs Pasting
- Bagging generally performs better than Pasting
- Bagging increases diversity via repetition
- Pasting is more deterministic

---

##  3. OOB Score (Out-of-Bag Score)
- Used in Bagging / Random Forest models
- ~36.8% data remains unused for each tree
- That unused data acts as validation (OOB samples)
- No need for separate validation dataset

---

##  4. Voting Ensemble

### 🔹 Hard Voting
- Majority voting (classification)
- Regression: average predictions

### 🔹 Soft Voting
- Weighted probability averaging
- Works better when models are well calibrated

---

#  Model Comparison

| Model | R² Score |
|------|----------|
| Linear Regression | 0.6721 |
| Ridge Regression | 0.6721 |
| Random Forest (Default) | -0.3531 |
| Random Forest (Tuned) | -0.3532 |

---

#  Random Forest Hyperparameter Tuning

##  GridSearchCV Result

- **Best R² Score:** -0.3529

### Best Parameters:
```python
{
 'max_depth': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 200
}
````

---

##  RandomizedSearchCV Result

* **Best R² Score:** -0.3532

### Best Parameters:

```python
{
 'max_depth': 20,
 'min_samples_leaf': 2,
 'min_samples_split': 7,
 'n_estimators': 285
}
```

---

# ⚠ Observations

* Linear Regression performed best, indicating a **linear dataset structure**
* Ridge Regression gave identical performance → low multicollinearity impact
* Random Forest failed even after tuning → mismatch between model and data
* Hyperparameter tuning did not improve performance significantly
* Ensemble methods depend heavily on dataset structure

---

#  Key Learnings

* Bagging reduces variance and improves stability
* Pasting removes duplication by sampling without replacement
* OOB score gives internal validation without test split
* Voting improves prediction stability
* GridSearchCV is slow but exhaustive
* RandomizedSearchCV is faster and more practical
* Model performance depends more on data than complexity

---

#  Final Conclusion

Linear Regression and Ridge Regression performed best for this dataset. Random Forest, even after tuning, failed to improve performance, confirming that the dataset is fundamentally linear and does not benefit from complex ensemble models.

---
##  Trade-off: Number of Trees vs Training Time

- Increasing `n_estimators` (number of trees) improves model stability and accuracy but significantly increases training time.
- Fewer trees reduce computation time but may lead to higher variance and less stable predictions.
- Therefore, a balance must be maintained between performance and computational efficiency.

---

##  AI Pro Tip

Random Forest is often used as a baseline model in Kaggle competitions. If you are unsure which model to choose, Random Forest is a strong starting point due to its robustness, ease of use, and ability to handle non-linear relationships without extensive preprocessing.

---
#  Final Insight

> Model selection should always be driven by data patterns, not algorithm complexity. Ensemble methods improve robustness, but cannot fix a fundamentally mismatched model assumption.


