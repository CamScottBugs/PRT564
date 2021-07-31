# Codes adapted from:
# Kane, F. 2017. Hands-On Data Science and Python Machine Learning. Packt Publishing.
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.inverse_transform


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 

from numpy import random, array, unique 

# Create synthetic income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(0, k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0),
                      random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

# create and inspect dataset
data = createClusteredData(100, 5)
print(data)

# initialise a K-Means object with 5 clusters
kmeans_param = 5
model = KMeans(n_clusters=kmeans_param)

# standardise all feature values in the dataset.
# standardised feature values will have 0 mean and unit std. deviation (i.e. standard normal distribution)
# standardisation often improves K-Means results.
scaler = StandardScaler()
model = model.fit(scaler.fit_transform(data))
# # activate the code below to show the effect of no standardisation:
# model = model.fit(data)

# inspect the cluster labels of each data point and centroids
centroids = model.cluster_centers_
kmeans_labels = model.labels_

# reverse centroid values to unstandardised feature values
centroids_invscaled = scaler.inverse_transform(centroids)

# visualise K-Means results
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=kmeans_labels.astype(float))
plt.scatter(centroids_invscaled[:,0], centroids_invscaled[:,1], marker="x", c="red")

# # activate the code below to show the effect of no standardisation:
# plt.scatter(centroids[:,0], centroids[:,1], marker="x", c="red")

plt.show()
