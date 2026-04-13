
# 📅 Day 8 Report – AI/ML Developer Track  
## 🏠 California Housing Price Prediction (End-to-End Regression Pipeline)

---

## 🔹 Objective

The objective of this task was to build a complete end-to-end regression pipeline using the California Housing dataset. The goal was to preprocess real-world data, apply feature engineering, train a Linear Regression model, and evaluate its performance using appropriate metrics and visualizations.

---

## 🔹 Dataset Overview

The California Housing dataset contains information about housing blocks, including features such as:

- Median Income (MedInc)  
- House Age (HouseAge)  
- Average Rooms (AveRooms)  
- Average Bedrooms (AveBedrms)  
- Population  
- Average Occupancy (AveOccup)  
- Latitude & Longitude (Location)  

**Target Variable:**  
- Median House Value (MedHouseVal)

---

## 🔹 Exploratory Data Analysis (EDA)

### 📊 Histogram Observations
- Most features such as Population, AveRooms, and AveOccup are right-skewed  
- MedInc shows moderate skewness with some high-value outliers  
- HouseAge is fairly distributed but shows a capped value  
- Population and AveOccup have large variation  
- MedHouseVal is nearly normal but capped at higher values  

---

### 📦 Box Plot Observations
- Population and AveOccup show extreme outliers  
- AveRooms and AveBedrms are right-skewed with high-value outliers  
- MedInc has moderate spread  
- HouseAge, Latitude, and Longitude are relatively stable  
- Presence of outliers can negatively affect model performance  

---

### 🔗 Correlation Analysis
- MedInc has strong positive correlation (0.68) with house prices  
- AveRooms and HouseAge have weak positive correlation  
- Latitude shows moderate negative correlation  
- Other features have weak or negligible correlation  

---

## 🔹 Data Preprocessing

### ✅ Log Transformation
Applied log transformation to reduce skewness:

```python
df['MedInc'] = np.log1p(df['MedInc'])
df['AveRooms'] = np.log1p(df['AveRooms'])
df['Population'] = np.log1p(df['Population'])
````

**Learning:**
Log transformation compresses large values and helps in making data more normally distributed, improving model learning.

---

### ✅ Clipping (Outlier Handling)

```python
df['AveOccup'] = np.clip(df['AveOccup'], 0, 10)
```

**Learning:**
Clipping limits extreme values within a range, preventing outliers from distorting model predictions.

---

## 🔹 Feature Engineering

Created new features to capture relationships:

```python
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']
```

**Learning:**
Feature engineering improves model performance by introducing meaningful relationships between variables.

---

## 🔹 Model Development

* Model Used: **Linear Regression**
* Data Split: 80% Training, 20% Testing
* Feature Scaling: Applied StandardScaler

---

## 🔹 Model Performance

* **Mean Absolute Error (MAE):** **45,570 $**
* **R² Score:** 0.67

### 📌 Interpretation:

* The model predicts house prices with good accuracy
* It explains approximately 67% of the variance in the dataset
* Remaining variance indicates presence of complex patterns

---

## 🔹 Residual Analysis

* Residuals are not randomly distributed
* A pattern is visible in the residual plot
* Error variance increases with predicted values
* Indicates presence of heteroscedasticity
* Suggests non-linear relationships in data

---

## 🔹 Key Learnings

* Data preprocessing is crucial for improving model performance
* Log transformation effectively reduces skewness
* Clipping helps manage extreme outliers
* Feature engineering enhances predictive capability
* Linear Regression has limitations with non-linear data
* Residual plots help identify model weaknesses

---

## 🔹 Conclusion

This project successfully demonstrated the importance of preprocessing, feature engineering, and evaluation in building a regression model. While Linear Regression provided good results, further improvements can be achieved using more advanced models to capture non-linear patterns.

---

## 🔹 Final Outcome

* Built a complete end-to-end regression pipeline
* Improved model performance through data transformation
* Gained hands-on experience with real-world data

---

## 🚀 Final Note

This task enhanced my understanding of practical machine learning workflows and strengthened my ability to build and evaluate predictive models effectively.


