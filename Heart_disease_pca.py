import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

heart = pd.read_csv(r"C:\Users\Mink\OneDrive\Documents\GitHub\Dataset-Save-Place\Heart Disease UCI Dataset\HeartDiseaseTrain-Test.csv")

y = heart["target"]

categorical_cols = ["sex", "chest_pain_type", "fasting_blood_sugar",
                    "rest_ecg", "exercise_induced_angina",
                    "slope", "vessels_colored_by_flourosopy", "thalassemia"]

heart_encoded = heart.copy()
for col in categorical_cols:
    le = LabelEncoder()
    heart_encoded[col] = le.fit_transform(heart_encoded[col])

X = heart_encoded.drop(columns=["target"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="target", palette="viridis", s=80, alpha=0.7)
plt.title("Heart Disease PCA (2D)", fontsize=14)
plt.grid(True)
plt.savefig("heart_pca_scatter.png", dpi=300)
plt.close()

pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_

plt.figure(figsize=(8,4))
sns.lineplot(x=range(1, len(explained_var)+1), y=explained_var, marker="o")
plt.title("Heart Disease - Explained Variance per Component", fontsize=14)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.savefig("heart_pca_scree.png", dpi=300)
plt.close()