import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("Loading CIFAR-10...")
(X,_),_ = cifar10.load_data()

# Flatten & normalize
X = X.reshape(len(X), -1).astype(float) / 255.0

# Use subset for speed
X = X[:5000]

print("Running PCA 2D...")
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

# Save explained variance
with open("pca2d_log.txt","w") as f:
    f.write("PCA 2D Explained Variance Ratio:\n")
    f.write(str(pca.explained_variance_ratio_))
    f.write("\nTotal explained variance: ")
    f.write(str(sum(pca.explained_variance_ratio_)))

# Plot & save figure
plt.figure(figsize=(7,6))
plt.scatter(X2[:,0], X2[:,1], s=4)
plt.title("CIFAR-10 PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca2d.png", dpi=300)
plt.close()

print("Done.")
print("Saved: pca2d.png and pca2d_log.txt")
