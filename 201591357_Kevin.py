# COMP 527 - DATA MINING AND VISUALIZATION - CA 2
# KEVIN JOHN MATHEW
# STUDENT ID : 201591357

import numpy as np
from matplotlib import pyplot as plt
import random 
import warnings

# Loading the dataset using loadtxt function
df = np.loadtxt('dataset', usecols=range(1,301))
sample_nos = df.shape[0]
nComponents = 2
max_iterations = 100
warnings.filterwarnings("ignore")

def distance(x, y):
    return np.sqrt(np.sum((x-y)**2))

def calc_centroids(data,k):
    centroids = data[np.random.choice(sample_nos, k, replace=False)]
    return centroids

def calc_distance(data,k,centers):
     distance_values = np.zeros((sample_nos,k))
     distance_values = [[((dataset - center) ** 2).sum() for center in centers] for dataset in data]
     return distance_values


# ---------------------- k-means function -------------------- #
def kmeans (data, k, max_iterations=100):
    # set the random seed
    np.random.seed(42)

    # Initialize an array to store the index of the closest centroid for each data point
    closest = np.zeros(sample_nos).astype(int)
    centroids = calc_centroids(data,k)

    while True:
        # Loop until convergence or until reaching the maximum number of iterations
        for i in range(max_iterations):
            distance_values = calc_distance(data,k,centroids)
             # Store the index of the closest centroid for each data point
            prev_closest = closest.copy()
            closest = np.argmin(distance_values, axis=1)

            # Update the centroids by computing the mean of the data points assigned to each cluster
            for i in range(k):
                centroids[i,:] = data[closest == i].mean(axis = 0)
            if all(closest == prev_closest):
                    break
        return closest, centroids


# function to plot the clusters along with centroids
def plot(principal,closest, centers):
    plt.scatter(principal[:, 0], principal[:, 1], c=closest, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()



# ------------- PCA function for dimensionality reduction ----------- #
def pca_conversion(data, nComponents):
   # Center the data by subtracting the mean along each column
    mean = np.mean(data, axis=0)
    data = data - mean

    # Calculate the covariance matrix of the centered data
    covariance_matrix = np.cov(data.T)

    # Calculate the eigenvectors and eigenvalues of the covariance matrix
    eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)

    sorted_eigenpairs = [(np.abs (eigenValues [i]), eigenVectors[:, i]) for i in range(len(eigenValues))]
    sorted_eigenpairs.sort(reverse=True)
    # Select the top nComponents eigenvectors as the principal components
    selected_eigenvectors = np.array([sorted_eigenpairs [i][1] for i in range (nComponents)])
    pca_dataset = np. dot (data, selected_eigenvectors.T)
    return pca_dataset


# ---------------------- k-means++ function -------------------- #
def k_means_plus_plus_clustering(data, k, max_iterations=100):
    d = data.shape[1]    # Dimension of data points

    # Step 1: Select the first cluster representative randomly
    centroids = np.zeros((k, d))
    index = np.random.choice(sample_nos)
    centroids[0] = data[index]

    # Step 2: Select the remaining cluster representatives using the k-means++ algorithm
    for j in range(1, k):
        distances = np.zeros(sample_nos)
        for i in range(sample_nos):
            distances[i] = np.min(np.linalg.norm(data[i] - centroids[:j], axis=1))
        probabilities = distances / np.sum(distances)
        index = np.random.choice(sample_nos, p=probabilities)
        centroids[j] = data[index]

    # Step 3: Assign each data point to the closest cluster representative
    clusters = np.zeros(sample_nos)
    for i in range(sample_nos):
        distances = np.linalg.norm(data[i] - centroids, axis=1)
        clusters[i] = np.argmin(distances)

    # Step 4: Update cluster representatives to the centroid of the data points in each cluster
    for j in range(k):
        centroids[j] = np.mean(data[clusters == j], axis=0)

    for i in range(max_iterations):
    # Step 5: Repeat steps 3 and 4 until convergence
        while True:
            prev_clusters = np.copy(clusters)
            for i in range(sample_nos):
                distances = np.linalg.norm(data[i] - centroids, axis=1)
                clusters[i] = np.argmin(distances)
            for j in range(k):
                centroids[j] = np.mean(data[clusters == j], axis=0)
            if np.array_equal(clusters, prev_clusters):
                break

    return clusters, centroids


# PLOTTING K-MEAN++ CLUSTERS
# max_iterations = 100
# nComponents = 2
# pca_data = pca_conversion (df, nComponents)
# labels, centers = k_means_plus_plus_clustering(pca_data, 5)
# print(labels)
# print(centers)
# plot(pca_data, labels,centers)


# ---------------------bisecting-k-means------------------------- #
def bisecting_kmeans(data, k, max_iterations=100):
    clusters = [data]
    sil_scores = []
    for i in range(k-1):
        max_score = -1
        for j in range(len(clusters)):
            curr_cluster = clusters[j]
            closest, centroids = kmeans(curr_cluster, 2, max_iterations)
            cluster_1 = curr_cluster[closest == 0]
            cluster_2 = curr_cluster[closest == 1]
            clusters[j] = cluster_1
            clusters.append(cluster_2)

            s1 = np.mean([np.mean(np.linalg.norm(cluster_1 - np.mean(cluster_1, axis=0), axis=1))])
            s2 = np.mean([np.mean(np.linalg.norm(cluster_2 - np.mean(cluster_2, axis=0), axis=1))])
            sil_score = (s2 - s1) / max(s1, s2)
            sil_scores.append(sil_score)

            if sil_score > max_score:
                max_score = sil_score
                best_cluster = cluster_1, cluster_2

        clusters.remove(best_cluster[0])
        clusters.remove(best_cluster[1])
        clusters.append(best_cluster[0])
        clusters.append(best_cluster[1])
    
    # Compute the final centroids
    final_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = [data[j] for j in range(data.shape[0]) if clusters[j] == i]
        final_centroids[i] = np.mean(points, axis=0)
    
    return clusters, final_centroids


# PLOTTING BISECTIONAL CLUSTERS
# max_iterations = 100
# nComponents = 2
# pca_data = pca_conversion (df, nComponents)
# labels, centers = bisecting_kmeans(pca_data, 5, 100)
# print(labels)
# plot(pca_data, labels, centers)


def silhouette_score(data, labels):
    sample_len = len(data)
    k = len(np.unique(labels))

    # Calculate the average distance from each point to all other points in its cluster
    a = np.zeros(sample_len)
    for i in range(sample_len):
        cluster_val = labels[i]
        cluster_points = data[labels == cluster_val]
        a[i] = np.mean(np.linalg.norm(data[i] - cluster_points, axis=1))

    # Calculate the average distance from each point to all points in the nearest neighboring cluster
    b = np.zeros(sample_len)
    for i in range(sample_len):
        cluster_val = labels[i]
        other_clusters = [j for j in range(k) if j != cluster_val]
        min_distance = float('inf')
        for j in other_clusters:
            cluster_points = data[labels == j]
            distance = np.mean(np.linalg.norm(data[i] - cluster_points, axis=1))
            if distance < min_distance:
                min_distance = distance
        b[i] = min_distance

    # Calculate the Silhouette coefficient for each point
    s = np.zeros(sample_len)
    for i in range(sample_len):
        s[i] = (b[i] - a[i]) / max(a[i], b[i])

    # Calculate the mean Silhouette coefficient for all points
    silhouette_coef = np.mean(s)

    return silhouette_coef

# Compute the Silhouette coefficient for each value of k
silhouette_kmean = []
silhouette_kmeanpp = []

for k in range(1, 10):
    labels, _ = kmeans(df, k)
    score = silhouette_score(df, labels)
    silhouette_kmean.append(score)

for k in range(1, 10):
    labels, _ = k_means_plus_plus_clustering(df, k)
    score = silhouette_score(df, labels)
    silhouette_kmeanpp.append(score)


# Plot the Silhouette coefficient for each value of k
plt.plot(range(1, 10), silhouette_kmean)
plt.title('Silhouette Graph for K-means')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette coefficient")
plt.show()

plt.plot(range(1, 10), silhouette_kmeanpp)
plt.title('Silhouette Graph for K-mean++')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette coefficient")
plt.show()

plt.title('Silhouette Graph Comparison')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette coefficient")
plt.plot(range(1, 10), silhouette_kmean, color='r', label='k-mean')
plt.plot(range(1, 10), silhouette_kmeanpp, label='k-mean++')
plt.legend()
plt.show()




