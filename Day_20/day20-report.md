
#  Day 20 Report: End-to-End ML Pipeline & Deployment

##  Objective
The objective of Day 20 was to transition from a research-based ML model to a **production-ready service** by:
- Building a complete ML pipeline
- Serializing the model
- Deploying it via a Flask API
- Simulating real-world integration

---

##  1. Model Development & Pipeline

An end-to-end pipeline was created using `sklearn.pipeline.Pipeline` to ensure consistent preprocessing and prediction.

###  Pipeline Components:
- **StandardScaler** → Feature scaling  
- **RandomForestRegressor** → Model training  

###  Benefits:
- Prevents data leakage  
- Ensures consistent transformations  
- Simplifies deployment  

---

##  2. Dataset Used
- California Housing Dataset  
- 8 Features:
  - MedInc  
  - HouseAge  
  - AveRooms  
  - AveBedrms  
  - Population  
  - AveOccup  
  - Latitude  
  - Longitude  

---

##  3. Model Serialization

The trained pipeline was saved as:

```

production_model.pkl

```

###  Why Serialization?
- Enables reuse without retraining  
- Allows deployment in production systems  
- Acts as the “brain” of the application  

---

##  4. Flask API Development

A REST API was built using Flask to serve predictions.

### 🔹 Endpoints:

####  `GET /`
- Health check  
- Returns service status, version, and required features  

####  `POST /predict`
- Accepts JSON input  
- Returns prediction (in 100k and USD format)  

####  `POST /batch_predict`
- Accepts multiple records  
- Returns batch predictions  

---

##  5. Integration Testing

Testing was performed using a separate Python script (`test_api.py`) to simulate backend communication.

---

### 🔹 Test 1: Health Check

```

GET /
Status: 200

````

✔ API is running successfully  

---

### 🔹 Test 2: Single Prediction

**Input:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
````

**Output:**

```json
{
  "prediction_100k": 4.2738,
  "prediction_usd": 427375.69
}
```

---

### 🔹 Test 3: Affordable Inland House

**Output:**

```json
{
  "prediction_100k": 0.7585,
  "prediction_usd": 75848.14
}
```

✔ Model captures real-world variation

---

### 🔹 Test 4: Batch Prediction

* Processed multiple inputs
* Returned structured results

---

### 🔹 Test 5: Error Handling

**Missing Features → Status 400**

```json
{
  "error": "Missing required features"
}
```

✔ Robust validation implemented

---

##  6. Reflection

Saving a Pipeline object is better than saving the scaler and model separately because it ensures that preprocessing and prediction are performed in a consistent and fixed sequence.

If saved separately:

* Risk of missing transformations
* Incorrect predictions possible
* Manual handling required

With Pipeline:

* Automatic preprocessing
* No risk of skipping steps
* Single file deployment
* Prevents data leakage

---

##  7. Integration Perspective

This system is designed for real-world integration:

* Backend (Node.js) → sends JSON data
* Flask API → processes and predicts
* Dashboard → visualizes results
* DevOps → deploys using containers

---

##  8. Completion Checklist

* ✔ Created sklearn Pipeline (Scaler + Model)
* ✔ Serialized model into `.pkl` file
* ✔ Built Flask API with `/predict` endpoint
* ✔ Tested API using POST requests
* ✔ Implemented batch prediction
* ✔ Added error handling

---

## 🚀 Conclusion

This project demonstrates how a Machine Learning model can be transformed into a **production-ready service** capable of real-time predictions and system integration.

It highlights the importance of:

* Pipeline consistency
* API-based deployment
* Cross-team integration
