# Reference: Brownlee, J. 2020. Basics of Linear Algebra for Machine Learning. pp. 146 - 149

# Principal Component Analysis (PCA) from scratch.

from numpy import array, mean, cov
from numpy.linalg import eig

# define a sample matrix that represents data
A = array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print("Original matrix:\n", A)

# calculate the mean of each column of the original matrix
M = mean(A, axis=0)
print("Means of column:\n", M)

# center columns by subtracting column means
C = A - M
print("Centered column:\n", C)

# calculate covariance matrix of centered matrix
V = cov(C.T)
print("Covariance matrix of centered matrix:\n", V)

# factorise covariance matrix
values, vectors = eig(V)
print("Eigenvalues (singular values):\n", values)
print("Eigenvectors (principal components):\n", vectors)

# project data
P = vectors.T.dot(C.T)
print("Original matrix with reduced dimensionality:\n", P.T)