import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode


# a function that maps K-Means Cluster labels to actual class labels
def map_cluster_to_class(clus, cls, k):
        labels = np.zeros_like(clus)
        for i in range(k):
                mask = (clus == i)
                labels[mask] = mode(cls[mask])[0]
        return labels

# define dataset variable names
wine_attr = ["Class",
                   "Alcohol",
                   "Malic acid",
                   "Ash",
                   "Alcalinity of ash",
                   "Magnesium",
                   "Total phenols",
                   "Flavanoids",
                   "Nonflavanoid phenols",
                   "Proanthosyanins",
                   "Color intensity",
                   "Hue",
                   "OD280/OD315",
                   "Proline"]

# read dataset
df = pd.read_csv("data/wine.data", names=wine_attr, header=None)

# reposition the "Class" column as the last column
df = df[[c for c in df if c not in ["Class"]] + ["Class"]]   
# print(df.info())
# print(df.describe())


# build K-Means Clustering model with 3 clusters
kmeans_param = 3
model = KMeans(n_clusters=kmeans_param, max_iter=100, random_state=0)
clusters = model.fit_predict(df.iloc[:,:-1])

# map cluster labels to class labels
clusters = map_cluster_to_class(clusters, df.Class, kmeans_param)

# # compare actual class labels, raw K-Means labels, and masked labels
class_labels = df.Class.values
print("True class labels:\n", class_labels)
print("K-Means cluster labels (raw)\n", model.labels_)
print("K-Means cluster labels (mapped to class)\n", clusters)

# print the accuracy of K-Means Clustering
print("Overall K-Means accuracy %.2f%%" % (accuracy_score(class_labels, clusters) * 100))

# visualise true class labels vs. K-Means clusters
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].scatter(df["Alcohol"], df["OD280/OD315"], c=class_labels, cmap="jet")
ax[1].scatter(df["Alcohol"], df["OD280/OD315"], c=clusters, cmap="jet")
ax[0].set_title("Actual", fontsize=18)
ax[1].set_title("KMeans", fontsize=18)
plt.show()

