from tensorflow.keras.datasets import cifar10
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

(X,_),_ = cifar10.load_data()
X = X.reshape(len(X), -1).astype(float) / 255.0
X = X[:3000]

pca = PCA(n_components=50)
X50 = pca.fit_transform(X)

km = KMeans(n_clusters=10, n_init=10, random_state=0)
labels = km.fit_predict(X50)

sil = silhouette_score(X50, labels)

print("PCA-50 Silhouette:", sil)
