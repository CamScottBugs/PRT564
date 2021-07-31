# Codes adapted from the following resources:
# Boschetti and Massaron. (2016). Python Data Science Essentials, 2nd Edn. Packt Publishing.
# https://stackabuse.com/generating-synthetic-data-with-numpy-and-scikit-learn/
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html

import numpy as np 
import matplotlib.pyplot as plt
import sklearn.datasets as dt

from sklearn.cluster import KMeans

# generate synthetic data points resembling four distinct, concentrated groups
N_samples = 2000
dataset_blobs, labels_blobs = dt.make_blobs(n_samples=N_samples, centers=4, cluster_std=0.4, random_state=0)

# build a KMeans model with 4 clusters
K_param = 4
kmeans = KMeans(n_clusters=K_param).fit(dataset_blobs)
kmeans_labels = kmeans.labels_

# # visualise the dataset with a scatterplot
# plt.scatter(dataset_blobs[:,0], dataset_blobs[:,1], c=labels_blobs, alpha=0.8, s=64, edgecolors="white")
# plt.show()

# visualise K-means clustering results
plt.scatter(dataset_blobs[:,0], dataset_blobs[:,1], c=kmeans_labels, alpha=0.8, s=64, edgecolors="black")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            marker="s", s=100, c=np.unique(kmeans_labels), edgecolors="black")
plt.show()
            