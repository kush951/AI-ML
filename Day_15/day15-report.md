# Day 15 Report: Unsupervised Learning & K-Means Clustering

## 🔗Objective
The objective of this experiment is to understand and implement **K-Means Clustering**, an unsupervised learning algorithm used to discover hidden patterns in data without labeled outputs.

---

## 🔗 Introduction

Unlike supervised learning, clustering does not use predefined labels. Instead, it groups data points based on similarity. K-Means works by minimizing the distance between data points and their respective cluster centers (centroids).

---

## 🔗 Methodology

### 1. Data Preparation
- Used synthetic dataset (via `make_blobs`) and real dataset (CGPA vs IQ).
- Selected relevant numerical features.
- Applied **StandardScaler** to normalize data (K-Means is extremely **sensitive** to Scaling.).

---

### 2. Elbow Method (Finding Optimal K)

- Ran K-Means for K = 1 to 10  
- Calculated WCSS (Within-Cluster Sum of Squares)  
- Plotted graph to identify "elbow point"

 **Observation:**  
The graph showed a sharp bend at **K = 5**, indicating the optimal number of clusters.

---

### 3. K-Means Clustering

- Applied K-Means with:
  - `n_clusters = 5`
  - `init = 'k-means++'`
- Predicted cluster labels.
- Visualized clusters using scatter plots.

---

### 4. Cluster Visualization

- Plotted clusters with distinct colors  .
- Highlighted centroids using marker 'X' . 
- Observed clear separation and compact grouping  .

---

## 🔗 Stability Test

### Experiment:
Ran K-Means 3 times with:
- Random initialization (different seeds).
- K-Means++ initialization.

### Observation:
- Random initialization produced **different cluster centers**  .
- K-Means++ produced **stable and consistent results**.

---

## 🔗 Concept Insight

K-Means minimizes the following:

WCSS = Σ Σ ||x - μ||²

- Lower WCSS → better clustering.
- K-Means++ helps reach lower WCSS consistently.

---

## 🔗 Importance of Scaling

K-Means is sensitive to feature scale:

- Large-scale features dominate distance calculation.  
- Smaller features get ignored.  

### Solution:
Used **StandardScaler**:
- Mean = 0  
- Std Dev = 1  

 Ensures fair contribution of all features  

---

## 🔗 Importance of `k-means++`

- Smart centroid initialization.  
- Prevents poor clustering.  
- Improves convergence speed.  
- Produces stable results.  

---
## 🔗 3D Clustering
To capture more complex relationships, clustering was extended to **3D space** by adding an additional feature.

---
## 🔗 Reflection (MeetMux Use Case)

Clustering helps identify behavioral user segments.  

If a cluster represents users who:
- Attend late-night events.  
- Prefer high-intensity sports.  

Then Mux AI Glass can:
- Recommend night-friendly features (low-light capture, navigation).  
- Suggest sports-related insights (performance tracking, stats).  
- Send timely notifications before events.  
- Improve matchmaking with similar users.  

 This shifts the system from **generic to behavior-based recommendations**

---

## 🔗 Results

- Successfully identified optimal K = 5  
- Generated well-separated clusters  
- Achieved stable clustering using K-Means++  
- Demonstrated importance of scaling  

---

## 🔗 Final Conclusion

K-Means Clustering effectively grouped users into meaningful segments based on similarity. The use of **k-means++ initialization and feature scaling** significantly improved clustering performance, stability, and interpretability.

---

## 🔗 Key Takeaway

K-Means is a powerful unsupervised learning technique that, when combined with proper initialization and scaling, can uncover valuable insights for real-world applications like personalized recommendations.

