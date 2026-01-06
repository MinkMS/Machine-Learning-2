import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

(X,_),_ = cifar10.load_data()
X = X.reshape(len(X), -1).astype(float) / 255.0
X = X[:3000]

idx = np.random.choice(X.shape[1], 100, replace=False)
Xsub = X[:, idx]

km = KMeans(n_clusters=10, n_init=10, random_state=0)
labels = km.fit_predict(Xsub)

sil = silhouette_score(Xsub, labels)

print("Random Subspace Silhouette:", sil)
