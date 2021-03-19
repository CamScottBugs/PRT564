from numpy import array
from numpy.linalg import norm

#define the vector
a = array([1,2,3])

#L-1 Norm
l1 = norm(a, 1)
print(l1)


b = array([2,3])
l1a = norm(b, 1)
print(l1a)
