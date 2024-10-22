# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:09:51 2024

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Menghasilkan data sampel
X = np.random.rand(100, 2)

# Membuat instance KMeans dengan 3 klaster
kmeans = KMeans(n_clusters=3, random_state=0)

# Melatih model dengan data
kmeans.fit(X)

# Mendapatkan label klaster
y_kmeans = kmeans.predict(X)

# Visualisasi klaster
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()