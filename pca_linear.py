# Codes are adapted from: 
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.decomposition import PCA

# Part I: Visualise eigenvalues and eigenvectors as vectors

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0,
                      shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    

# initialise a random number generator
rng = np.random.RandomState(1)

# create two matrices
A = rng.rand(2, 2)
B = rng.rand(2, 200)

# matrix A shape: 2 rows x 2 columns
print(A.shape)
print("Matrix A:\n", A)

# matrix B shape: 2 rows x 200 columns
print(B.shape)
print("Matrix B:\n", B)

# create the dataset as the dot product of matrix A and B
# the result is a new matrix shape with 2 rows x 200 columns
X = np.dot(A, B)
print(X.shape)

# convert matrix X to 200 rows x 2 columns
X = X.T 
print(X.shape) 

# visualise a scatterplot
plt.scatter(X[:,0], X[:,1])
plt.axis('equal')
plt.show()

# create a PCA object
pca = PCA(n_components=2)

# fit PCA on the dataset
pca.fit(X)

# show eigenvalues and eigenvectors
eig_values = pca.explained_variance_
eig_vectors = pca.components_
print("Eigenvalues (singular values):\n", eig_values)
print("Eigenvectors (principal components):\n", eig_vectors)


# plot eigenvectors and eigenvalues as vectors over the input data points
plt.scatter(X[:,0], X[:,1], alpha=0.2)
for magnitude, vector in zip(eig_values, eig_vectors):
    v = vector * -1.5 * np.sqrt(magnitude)
    print(magnitude, vector, v)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()




# # Part II: Using PCA for dimensionality reduction

# # rebuild PCA with only 1 component
# pca = PCA(n_components=1)

# # refit PCA on the dataset
# pca.fit(X)

# # get transformed dataset with reduced dimensionality
# X_pca = pca.transform(X)
# print("Original data shape:                           ", X.shape)
# print("Transformed shape with reduced dimensionality: ", X_pca.shape)

# # inverse transform the reduced data
# # and plot it along the original data
# X_pca_inv = pca.inverse_transform(X_pca)
# plt.scatter(X[:,0], X[:,1], alpha=0.2)
# plt.scatter(X_pca_inv[:,0], X_pca_inv[:,1], alpha=0.8)
# plt.axis('equal')
# plt.show()