# Codes adapted from:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

# a function that maps K-Means Cluster labels to actual class labels
def map_cluster_to_class(clus, cls, k):
        labels = np.zeros_like(clus)
        for i in range(k):
                mask = (clus == i)
                labels[mask] = mode(cls[mask])[0]
        return labels


# load the digits dataset from Scikit-Learn
digits = load_digits()

# # inspect the dataset
print(digits.data.shape)
# print(digits)

# # inspect the 1st digit image
# plt.gray() 
# plt.matshow(digits.images[0]) 
# plt.show() 

# build k-means with 10 clusters (corresponding to digits 0, 1, 2, ..., 9)
kmeans_param = 10
model = KMeans(n_clusters=kmeans_param, random_state=0)
clusters = model.fit_predict(digits.data)
# print(model.cluster_centers_.shape)   # the result is 10 clusters with 64 dimensions

# visualise the cluster centers (i.e. representative images of each digit)
fig, ax = plt.subplots(2, 5, figsize=(8,3))
centroids = model.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centroids):
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