# Reference: Brownlee, J. 2020. Basics of Linear Algebra for Machine Learning. pp. 146 - 149

# Principal Component Analysis (PCA) with Scikit-Learn

from numpy import array
from sklearn.decomposition import PCA

# define matrix representing data
A = array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print("Original matrix:\n", A)

# create PCA object with 2 components to keep
pca = PCA(2)
# try this and observe: pca = PCA(1)

# fit PCA on the dataset
pca.fit(A)

# show eigenvalues and eigenvectors
print("Eigenvalues (singular values):\n", pca.explained_variance_)
print("Eigenvectors (principal components):\n", pca.components_)

# project the original data into the learned subspace
B = pca.transform(A)
print("Original matrix with reduced dimensionality:\n", B)

