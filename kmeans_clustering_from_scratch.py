import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

dataBase = pd.read_csv('Dry_Bean.csv')
encoded_database = pd.get_dummies(dataBase, columns=["Class"])
encoded_database = encoded_database.astype(int)
clustering_completed = False
n_components = 2  
pca = PCA(n_components=n_components)
pca.fit(encoded_database)
reduced_d = pca.transform(encoded_database)
np.around(reduced_d, 3)

K=2
def kMeans_init_centroids(X, K):

    randidx = np.random.permutation(X.shape[0])

    centroids = X[randidx[:K]]
    
    return centroids

def find_closest_centroids(X, centroids):
    global sumdistance
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = [] 
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j]) 
            distance.append(norm_ij)
            sumdistance = sum(distance)
        idx[i] = np.argmin(distance)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):   
        points = X[idx == k] 
        centroids[k] =  np.mean(points, axis = 0)
        
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    global sumdistance
    m = X.shape[0]
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
        centroid_difference = np.linalg.norm(centroids - previous_centroids)
        
        if centroid_difference < 0.001:
            print("Converged at iteration", i)
            break
        previous_centroids = centroids.copy()
    return centroids, idx

initial_centroids = kMeans_init_centroids(reduced_d, K)
max_iters = 100

centroids, idx = run_kMeans(reduced_d, initial_centroids, max_iters)



plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'orange', 'purple']  # Add more colors as needed

for i in range(K):
    cluster_points = reduced_d[idx == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}')

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Results')
plt.legend()
plt.show()
