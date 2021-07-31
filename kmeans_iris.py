# Codes adapted from:
# https://predictivehacks.com/k-means-elbow-method-code-for-python/


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn import datasets

# load dataset
iris = datasets.load_iris()
# print(type(iris))

# explore input variables
df = pd.DataFrame(iris["data"])
print(df.head())
print(df.info())
print(df.describe())

# explore target variable
df_target = pd.DataFrame(iris["target"])
print(df_target.head())
print(df_target.info())
print(df_target.describe())

# find the optimal number of clusters K
# using the Elbow Method visualisation
# inertia = within-cluster sum of squares
distortions = []
kmeans_params = range(1, 10)
for k in kmeans_params:
    model = KMeans(n_clusters=k)
    model.fit(df)
    distortions.append(model.inertia_)
plt.figure(figsize=(16,8))
plt.plot(kmeans_params, distortions, "bx-")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("The Elbow Method showing the optimal k")
plt.show()

# update KMeans model using the optimal K
model = KMeans(n_clusters=3)
model.fit(df)
df["predicted"] = model.predict(df)
df["target"] = df_target[0]

# compare actual cluster versus KMeans predicted cluster
fig, ax = plt.subplots(1, 2, figsize=(16,8))
ax[0].scatter(df[0], df[1], c=df["target"], cmap=plt.cm.Set1)
ax[1].scatter(df[0], df[1], c=df["predicted"], cmap=plt.cm.Set1)
ax[0].set_title("Actual", fontsize=18)
ax[1].set_title("KMeans", fontsize=18)
plt.show()
