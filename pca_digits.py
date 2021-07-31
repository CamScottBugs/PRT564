# Parts of these codes are adapted from:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

# Part I: explore PCA components of the dataset

# load digits dataset
digits = load_digits()

# project the dataset onto 2-dimensional subspace
pca = PCA(n_components=2)  # project from 64 to 2 dimensions
projected_2 = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected_2.shape)

# visualise digits over 2 PCA components
plt.scatter(projected_2[:, 0], projected_2[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

# compare the cumulative explained variance versus number of PCA components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# find number of PCA components that explain 90% of the variance
pca = PCA(0.90).fit(digits.data)
print("90%% variance is explained by: %.d components." % pca.n_components_)

# find number of PCA components that explain 95% of the variance
pca = PCA(0.95).fit(digits.data)
print("95%% variance is explained by: %.d components." % pca.n_components_)

# find number of PCA components that explain 99% of the variance
pca = PCA(0.99).fit(digits.data)
print("99%% variance is explained by: %.d components." % pca.n_components_)


# Part II: K-Means Clustering performance with reduced dimensionality

# a function that maps K-Means Cluster labels to actual class labels
def map_cluster_to_class(clus, cls, k):
        labels = np.zeros_like(clus)
        for i in range(k):
                mask = (clus == i)
                labels[mask] = mode(cls[mask])[0]
        return labels

# rerun PCA with 21 components
pca = PCA(n_components=21)  # project from 64 to 21 dimensions
projected_21 = pca.fit_transform(digits.data)
print(projected_21.shape)


# build k-means with 10 clusters (corresponding to digits 0, 1, 2, ..., 9)
kmeans_param = 10
model = KMeans(n_clusters=kmeans_param, random_state=0)
clusters = model.fit_predict(projected_21)
centroids = model.cluster_centers_
print(centroids.shape) # the result is 10 clusters with 21 dimensions

# compare the original image vs. reduced dimensionality image
projected_21_inv = pca.inverse_transform(projected_21)
print(digits.data[0])
print(projected_21[0])
print(projected_21_inv[0])

# visualise the cluster centers (i.e. representative images of each digit)
centroids_inv = pca.inverse_transform(centroids)
centroids_inv = centroids_inv.reshape(10, 8, 8)
fig, ax = plt.subplots(2, 5, figsize=(8,3))
for axi, center in zip(ax.flat, centroids_inv):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation="nearest", cmap=plt.cm.binary)
plt.show()

# print the accuracy of KMeans clustering
labels = map_cluster_to_class(clusters, digits.target, kmeans_param)
print("Overall K-Means accuracy %.2f%%" % (accuracy_score(digits.target, labels)*100))

# visualise confusion matrix
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False,
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.show()