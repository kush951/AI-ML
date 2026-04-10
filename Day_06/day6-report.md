# Day 6 Report: Logistic Regression & Classification Metrics

---

# Objective

The objective of this experiment is to understand binary classification using Logistic Regression and evaluate model performance using confusion matrix and classification metrics.

---

#  SECTION 1: Basic Experiment (Sleep vs Coffee)

##  Output: Model Prediction

Model Prediction for Test Data: [0 1]
Actual: [0 1]

👉 The model correctly predicted both test samples.

---

##  Confusion Matrix

```
[[1 0]
 [0 1]]
```

👉 Interpretation:

* 1 True Negative (Fail correctly predicted)
* 1 True Positive (Pass correctly predicted)
* No errors

---

##  Classification Report
* Accuracy: **1.00 (100%)**
* Precision, Recall, F1-score: **All 1.00**

👉 Model performed perfectly (due to small dataset).

---

##  Feature Importance

* Hours_Sleep → **+0.6324 (Positive Impact)**
* Coffee_Cups → **-0.9365 (Negative Impact)**

👉 Interpretation:

* More sleep increases chances of passing
* More coffee decreases chances of passing

---

##  Prediction Experiment

* Input: [3 hours sleep, 7 coffees] → **FAIL ❌**
* Input: [8 hours sleep, 1 coffee] → **PASS ✅**

👉 Model behaves logically based on learned patterns.

---

##  Insight

> Sleep positively influences performance, while excessive coffee negatively affects outcomes.

---

#  SECTION 2: Breast Cancer Dataset (Real-World Application)

##  Confusion Matrix

```
[[70  1]
 [ 2 41]]
```

👉 Interpretation:

* 70 True Negatives (Correctly identified malignant cases)
* 41 True Positives (Correctly identified benign cases)
* 1 False Positive
* 2 False Negatives ⚠️

---

##  Classification Report

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| 0 (Malignant) | 0.97      | 0.99   | 0.98     |
| 1 (Benign)    | 0.98      | 0.95   | 0.96     |

👉 Overall Accuracy: **97%**

---

##  Critical Insight

* False Negatives = **2 cases** 
* These are dangerous because cancer cases were missed

👉 In medical systems:

> Minimizing False Negatives is more important than maximizing accuracy

---

##  Observations

* Model performs very well with high accuracy (97%)
* Strong ability to detect malignant tumors (high recall for class 0)
* Slight drop in recall for benign class
* Very few misclassifications

---

#  Final Conclusion

* Logistic Regression is effective for binary classification problems
* Simple datasets can show perfect accuracy but are not realistic
* Real-world datasets require proper evaluation using confusion matrix
* Error analysis is critical, especially in sensitive domains like healthcare

---

#  Final Learning

* Understood difference between regression and classification
* Learned importance of confusion matrix over accuracy
* Gained insight into feature importance and real-world decision-making

---

#  Final Statement

This experiment provided practical exposure to classification problems and highlighted the importance of error analysis and model evaluation in real-world AI applications.

---
