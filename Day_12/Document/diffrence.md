#  Difference Between Class Predictions and Class Probabilities

## 🔹 Class Predictions (`predict()`)

* Returns the **final predicted class label** (e.g., 0 or 1)
* Represents a **hard decision** made by the model
* Based on a threshold (usually **0.5** in binary classification)
* Does **not provide confidence level**

**Example:**

```python
model.predict(X_test_scaled)
```

Output:

```text
[1, 0, 1]
```

---

## 🔹 Class Probabilities (`predict_proba()`)

* Returns the **probability of each class**
* Represents a **soft decision (confidence score)**
* Helps understand how confident the model is
* Sum of probabilities for each sample = **1**

**Example:**

```python
model.predict_proba(X_test_scaled)
```

Output:

```text
[[0.10, 0.90],
 [0.85, 0.15],
 [0.30, 0.70]]
```

---

## 🔹 Key Differences

| Feature    | Class Prediction | Class Probability         |
| ---------- | ---------------- | ------------------------- |
| Output     | Final class (0/1) | Probability values        |
| Type       | Hard decision    | Soft decision             |
| Confidence |  Not shown       |  Shown                    |
| Use        | Final result     | Analysis & decision-making |


