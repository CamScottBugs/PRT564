from numpy import array

a = array([1,2,3])
b = array([1,2,3])

#dot prod= 1*1 + 2*2 + 3*3 = 14 
c = a.dot(b)
print(c)

d = b.dot(a)
print(d)

#machine leanring weighted sum ex:
data = array([1,2,3])
weights = array([0.5,0.8,1])

ws = data.dot(weights)
print(ws)
