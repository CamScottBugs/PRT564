from numpy import array, diag

M = array([
[1,2,3],
[1,2,3],
[1,2,3]
])


#step 1: extract the diagonl vector
d = diag(M)
print(d)

#step 2:  build the diagnol matric
D = diag(d)
print(D)