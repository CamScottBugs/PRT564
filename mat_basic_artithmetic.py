from numpy import array

A = array([
    [1,2,3],
    [4,5,6]
])

print("Matrixd A")
print(A)

B= array([
[1,2,3],
[4,5,6]
])

print("Matrix B")
print(B)

#add matric A and B
C = A+B
print(C)

#subtracts matrix C from A
E= A -C
print(" the resul of C form A:\n",E)

#multiply matrices ("HAdamard Product")
F = A * B
print("the result of mult A and B:\n",F)

#divide matrices 
G = A / B
print("the rult of A div B:\n",G)

