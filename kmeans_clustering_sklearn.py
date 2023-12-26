import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

dataBase = pd.read_csv('Dry_Bean.csv')
encoded_database = pd.get_dummies(dataBase, columns=["Class"])
encoded_database = encoded_database.astype(int)
n_components = 2  # Number of components to keep
pca = PCA(n_components=n_components)
pca.fit(encoded_database)
reduced_d = pca.transform(encoded_database)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)

kmeans.fit(reduced_d)

cluster_labels = kmeans.labels_

plt.scatter(reduced_d[:, 0], reduced_d[:, 1], c=cluster_labels, cmap='rainbow')
plt.title("PCA: Reduced Dimensions Scatter Plot")
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()