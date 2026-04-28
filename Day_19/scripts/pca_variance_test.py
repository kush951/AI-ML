
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from Day_19.scripts.pca_reduction import X

# 1. Run PCA without specifying components to see all
pca_full = PCA().fit(X)

# 2. Calculate Cumulative Variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 3. Find how many components are needed for 95% variance
n_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"Number of components needed to keep 95% info: {n_95}")

# 4. Plot
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()