# Codes adapted from the following resources:
# Boschetti and Massaron. (2016). Python Data Science Essentials, 2nd Edn. Packt Publishing.
# https://stackabuse.com/generating-synthetic-data-with-numpy-and-scikit-learn/
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html

import numpy as np 
import matplotlib.pyplot as plt
import sklearn.datasets as dt

from sklearn.cluster import KMeans

# generate synthetic data points resembling two-layered circles
N_samples = 2000
dataset_circles, labels_circles = dt.make_circles(n_samples=N_samples, noise=0.05, factor=0.3)

# build a KMeans model with 2 clusters
K_param = 2
kmeans = KMeans(n_clusters=K_param).fit(dataset_circles)
kmeans_labels = kmeans.labels_

# # visualise the dataset with a scatterplot
# plt.scatter(dataset_circles[:,0],dataset_circles[:,1], c=labels_circles, alpha=0.8, s=64, edgecolors="white")
# plt.show()

# visualise K-means clustering results
plt.scatter(dataset_circles[:,0],dataset_circles[:,1], c=kmeans_labels, alpha=0.8, s=64, edgecolors="black")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            marker="s", s=200, c=np.unique(kmeans_labels), edgecolors="black")
plt.show()
