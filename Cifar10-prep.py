import numpy as np
from tensorflow.keras.datasets import cifar10

print("Loading CIFAR-10...")

(X, y), _ = cifar10.load_data()

# Flatten 32×32×3 → 3072
X = X.reshape(len(X), -1).astype(float) / 255.0

print("Samples:", X.shape[0])
print("Features:", X.shape[1])
