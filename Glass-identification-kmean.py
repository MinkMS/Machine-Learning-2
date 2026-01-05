import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("===== GLASS DATASET =====")

df = pd.read_csv(r"C:\Users\Mink\OneDrive\Documents\GitHub\Dataset-Save-Place\Glass Identification\glass.csv")

# Drop label column
X = df.drop(columns=["Type"])

# Standardize
X = StandardScaler().fit_transform(X)

for k in range(2, 9):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=0)
    labels = km.fit_predict(X)

    inertia = km.inertia_
    sil = silhouette_score(X, labels)

    print(f"k={k:2d} | Inertia={inertia:10.2f} | Silhouette={sil:.4f}")
