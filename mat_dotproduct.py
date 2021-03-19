from numpy import array

A = array([
    [1,2],
    [3,4],
    [5,6]
    ])

B = array([
    [1,2],
    [3,4]
    ])

C= A.dot(B)
print("the dot product of two matrixes:\n",C)

# D= B.dot(A)
# print(D)
#Cannot produce as non-communative!

# How do we get 7?  [1,2] (from A) dot product [1,3]:  1*1+2*3=7
# How do we get 10?  [1,2] (from A) dot product [2,4]:  1*2 + 2*4 = 10

# C= A @ B # for Python 3.5 and up

#define vector
D = array([0.5, 0.5])

E = A.dot(D)
print(E)
