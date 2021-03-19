from numpy import array, tril, triu

M = array([
    [1,2,3],
    [1,2,3],
    [1,2,3]
])
print(M)

lower = tril(M)
print("lower triagular matrix:\n",lower)

upper = triu(M)
print("Upper triangular matix:\n",upper)
