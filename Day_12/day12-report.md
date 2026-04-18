# Day 12: Logistic Regression & Classification Metrics

---

## 🔗 Objective

The objective of this experiment is to understand **Binary Classification** using Logistic Regression and evaluate model performance using metrics like **Accuracy, Confusion Matrix, Precision, Recall, and F1-score**.

---

## 🔗 Dataset Used

* **Breast Cancer Wisconsin Dataset**
* Task: Predict whether a tumor is:

  * **0 → Benign**
  * **1 → Malignant**

---

## 🔗 Implementation Steps

### 1. Data Loading & Splitting

* Dataset loaded using `sklearn.datasets`
* Split into training and testing sets (80:20)

### 2. Feature Scaling

* Applied `StandardScaler`
* Important for Logistic Regression performance

### 3. Model Training

* Used `LogisticRegression`
* Trained on scaled training data

### 4. Prediction

* Predicted class labels using `predict()`

---

## 🔗 Model Performance

### ✔ Accuracy

* Achieved accuracy: **~96.5%**

---

### 🔗 Confusion Matrix

```
[[36  3]
 [ 1 74]]
```

#### Interpretation:

* **True Negatives (TN)** = 36
* **False Positives (FP)** = 3
* **False Negatives (FN)** = 1
* **True Positives (TP)** = 74

---

## 🔗 Observations

* The model performs **very well** with high accuracy.
* Most predictions are correct (diagonal values are high).
* Only **1 False Negative**, which is critical in medical cases.
* Few False Positives (3), which are less harmful than FN.
* Model is **reliable for classification tasks**.

---

## 🔗 Key Insight

* Accuracy alone is not enough in medical applications.
* **False Negatives must be minimized**, as they can miss actual disease cases.

---

## 🔗 Probability Analysis

Used:

```python
model.predict_proba(X_test_scaled)
```

### Observations:

* Model outputs **probabilities for each class**
* Each row sums to **1**
* Highest probability determines final class

---

## 🔗 Class Predictions vs Class Probabilities

| Feature    | Class Prediction  | Class Probability  |
| ---------- | ----------------- | ------------------ |
| Output     | Final class (0/1) | Probability values |
| Type       | Hard decision     | Soft decision      |
| Confidence | Not shown         | Shown              |
| Use        | Final result      | Decision analysis  |

---

## 🔗 Reflection

### Q: Which is worse: False Positive or False Negative?

> In a medical scenario like cancer detection, a **False Negative is far worse** than a False Positive.
>
> A False Negative means predicting that a patient is healthy when they actually have cancer. This can delay treatment and may lead to serious health risks or even death.
>
> A False Positive, although stressful, only leads to additional testing and does not pose a direct life threat.
>
> Therefore, minimizing False Negatives is critical in healthcare systems.

---

## 🔗 Conclusion

* Logistic Regression successfully performs **binary classification**.
* The model achieved **high accuracy and reliable predictions**.
* Confusion Matrix provided deeper insight into errors.
* Probability outputs helped understand **model confidence**.
* The model is effective but should be evaluated carefully in **critical applications like healthcare**.

---

## 🔗 Final Checklist

* ✔ Dataset loaded and scaled
* ✔ Logistic Regression model trained
* ✔ Accuracy > 90% achieved
* ✔ Confusion Matrix generated
* ✔ Probability analysis performed
* ✔ Difference between prediction and probability documented

---

## 🔗Final Remark

Logistic Regression is a **simple yet powerful baseline model** (i.e **Hello world**) for classification problems. It provides both predictions and probabilities, making it highly useful in real-world applications.
