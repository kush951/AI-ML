# 🚀 Day 4 – Pre-processing Techniques (AI/ML Track)

## 🔧 Technical Summary
Today I focused on data preprocessing, which is an important step in machine learning.

I implemented:
- Handling missing values using Pandas (Imputation)
- Filling Age using mean and Score/Salary using default/median values
- Feature scaling using MinMaxScaler
- Data visualization using Seaborn (Heatmap, Pairplot, Histogram, Boxplot)

---

## 🐞 Bug Log
**Issue:** Error while saving files  
`OSError: Cannot save file into a non-existent directory`

**Fix:**  
Created folder using:
```python
import os
os.makedirs("outputs", exist_ok=True)