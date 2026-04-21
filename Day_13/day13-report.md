# 🚀 Day 13 — Capstone Integration Report

## MeetMux · Mux Intelligence Prototype

**Track:** AI/ML Developer
**Project:** Event Networking Matcher

---

# 🔗 Objective

Build a production-ready **classification engine (Mux Intelligence)** that predicts **User Compatibility** for event networking.

The system demonstrates three core ML integration points:

* **Polynomial Features** → Capture non-linear interactions
* **Random Forest Classifier** → Perform final prediction
* **Confusion Matrix + Feature Importance** → Evaluate & explain decisions

Additionally, the system is extended with a **Flask API layer** for real-world deployment.

---

# 🔗 Dataset Used

| Property   | Details                                |
| ---------- | -------------------------------------- |
| Type       | Synthetic (programmatically generated) |
| Users      | 300 attendee profiles                  |
| Pairs      | 3,000 pairwise examples                |
| Match Rate | ~36.5%                                 |
| Test Split | 20% (600 samples)                      |
| Noise      | Gaussian noise (σ = 0.08)              |

---

## 🔗 Attendee Features

| Feature    | Type        | Description                        |
| ---------- | ----------- | ---------------------------------- |
| domain     | Categorical | AI/ML, Data Science, Backend, etc. |
| goal       | Categorical | Co-founder, Job, Hiring, Learning  |
| experience | Numeric     | Years (0–15)                       |
| comm_style | Categorical | Direct, Collaborative, Analytical  |
| location   | Categorical | City or Remote                     |

---

## 🔗 Pairwise Similarity Features

| Feature    | Weight | Logic                        |
| ---------- | ------ | ---------------------------- |
| domain_sim | 0.28   | Same domain → 1 else 0       |
| goal_sim   | 0.24   | Compatible → 1 else 0.2      |
| comm_sim   | 0.18   | Compatible → 1 else 0.15     |
| exp_sim    | 0.16   | `max(0, 1 - diff/15)`        |
| loc_sim    | 0.14   | Same=1, Remote=0.7, else=0.3 |

---

# 🔗 Implementation Steps

## 🔹 Step 1 — Data Generation (`data_generator.py`)

* Generated 300 users using `numpy.random`
* Created 3,000 user pairs
* Computed similarity scores
* Added Gaussian noise (σ = 0.08)
* Generated binary labels

---

## 🔹 Step 2 — Preprocessing (`preprocessor.py`)

* Applied:

  * `PolynomialFeatures(degree=2)`
  * `StandardScaler`

**Feature Expansion:**

* Original: **5 features**
* After Polynomial: **20 features**

Examples:

* `domain_sim × goal_sim`
* `exp_sim²`
* `comm_sim × loc_sim`

---

## 🔹 Step 3 — Model Training (`model.py`)

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
```

* Train Size: 2400
* Test Size: 600

---

## 🔹 Step 4 — Evaluation (`evaluator.py`)

* Generated:

  * Confusion Matrix
  * Classification Report
  * Evaluation plots

---

## 🔹 Step 5 — Live Prediction (`predictor.py`)

* Accepts two user profiles
* Outputs:

  * Compatibility score
  * Match/No Match
  * Top contributing features

---

#  🔗Model Performance

##  Classification Report

```text
Accuracy  : 90.3%
Precision : 85.2%
Recall    : 89.0%
F1 Score  : 87.1%
```

---

##  Confusion Matrix

```text
TP: 195   FN: 24
FP: 34    TN: 347
```

|                 | Predicted Match | Predicted No Match |
| --------------- | --------------- | ------------------ |
| Actual Match    | 195             | 24                 |
| Actual No Match | 34              | 347                |

---

##  🔗Interpretation

* Accuracy: Model correctly classifies **9/10 pairs**
* Precision: 85% match suggestions are correct
* Recall: Captures **89% real matches**
* Balanced performance with good generalization

---

#  🔗Feature Importance

| Rank | Feature               | Insight                     |
| ---- | --------------------- | --------------------------- |
| 1    | goal_sim              | Most important factor       |
| 2    | domain_sim × goal_sim | Strong interaction          |
| 3    | domain_sim            | Technical alignment         |
| 4    | comm_sim              | Communication compatibility |
| 5    | exp_sim               | Experience matters          |
| 6    | domain_sim²           | Strong domain boost         |
| 7    | exp_sim × comm_sim    | Learning compatibility      |
| 8    | loc_sim               | Less important              |
| 9    | goal_sim²             | Reinforcement               |
| 10   | comm_sim × loc_sim    | Minor                       |

---

# 🔗 API & Integration Layer

## ⚙️ Flask API (`app.py`)

### Endpoints:

### 🔹 POST `/predict`

* Input: two user profiles
* Output:

  * Compatibility score
  * Match/No Match
  * Top signals

---

### 🔹 GET `/feature-importance`

* Returns ranked feature importance

---

### 🔹 GET `/metrics`

* Returns:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix

---

##  🔗Integration Flow

```text
User Input → API → Preprocessing → Model → Prediction → Response
```

---

## 🔗 Integration Highlights

* Real-time predictions
* Modular architecture
* Explainable AI outputs
* Easily deployable

---

# 🔗Observations

1. Polynomial features improved accuracy from **~84% → 90.3%**
2. Model slightly favors **false positives over false negatives**
3. Recall is strong due to **class balancing**
4. Location has minimal impact
5. Experience importance increases via interactions

---

# 🔗 Key Insight Reflection

> Theory assumes clean labels. Reality is noisy.

* Added Gaussian noise to simulate real-world behavior
* Model learned patterns instead of memorizing
* Achieved robust and realistic performance

---

# 🔗Project Structure

```text
meetmux/
├── config.py
├── data_generator.py
├── preprocessor.py
├── model.py
├── evaluator.py
├── predictor.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔗Run Instructions

```bash
python main.py
```

Outputs:

* Model → `models/random_forest.joblib`
* Plot → `outputs/evaluation.png`

---

# 🔗Conclusion

* Built a complete **end-to-end ML system**
* Achieved strong performance (**90.3% accuracy**)
* Integrated **model + preprocessing + API**
* Enabled **real-time matchmaking + explainability**

---

# 🔗 Final Remark

Mux Intelligence transforms raw user data into meaningful connections — combining machine learning, system design, and real-world applicability.

This is not just a model.
It is a **deployable AI system**.

---
