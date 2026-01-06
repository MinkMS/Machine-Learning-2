from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

(X,_),_ = cifar10.load_data()
X = X.reshape(len(X), -1).astype(float) / 255.0
X = X[:3000]     # lấy 3000 mẫu cho nhẹ

km = KMeans(n_clusters=10, n_init=10, random_state=0)
labels = km.fit_predict(X)

sil = silhouette_score(X, labels)

print("High-dim Silhouette:", sil)
