from numpy import array

#define Othogonal Matrix
Q = array([
    [1,0],
    [0,-1]
])

print(Q)

# Identity equivalence
print("Identitty equivaled:")
I =Q.dot(Q,T)
print(I)

#invers equivalec
print("inverse equiv:")
V=inv(Q)
print("printing inverse euivence:")
print(V)

