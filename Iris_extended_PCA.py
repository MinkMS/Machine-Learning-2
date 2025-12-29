import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

iris_ext = pd.read_csv(r"C:\Users\Mink\OneDrive\Documents\GitHub\Dataset-Save-Place\Iris Extended\iris_extended.csv")

y = iris_ext["species"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = iris_ext.drop(columns=["species", "soil_type"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["species"] = y

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="species", palette="viridis", s=80, alpha=0.7)
plt.title("Iris Extended PCA (2D)", fontsize=14)
plt.grid(True)
plt.savefig("iris_extended_pca_scatter.png", dpi=300)
plt.close()

pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_

plt.figure(figsize=(8,4))
sns.lineplot(x=range(1, len(explained_var)+1), y=explained_var, marker="o")
plt.title("Iris Extended - Explained Variance per Component", fontsize=14)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.savefig("iris_extended_pca_scree.png", dpi=300)
plt.close()