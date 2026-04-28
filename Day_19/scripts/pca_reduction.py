import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

#Load Dataset (64 features)
digits = load_digits()
X = digits.data   # shape = (1797, 64)
y = digits.target

#Standardization (IMPORTANT)

# PCA is variance-based → scaling improves results
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Initialize PCA (64 → 2)

pca = PCA(n_components=2)

#Fit and Transform
X_reduced = pca.fit_transform(X_scaled)

# 5. Explained Variance
variance_retained = sum(pca.explained_variance_ratio_) * 100
print(f"Variance retained by 2 components: {variance_retained:.2f}%")

#Visualization
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=y,
    cmap='viridis',
    s=20
)

plt.colorbar(scatter, label='Digit Class')
plt.title("Digits Dataset Projected into 2D Space via PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()