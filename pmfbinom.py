from scipy.stats import binom
from matplotlib import pyplot

p = 0.3
k = 100

#define the binomial distribution
dist = binom(k, p)

#print("Prob of each successful outcome:")
#for n in range(10, 110, 10):
#   print("Prob of %d successes: %.3f%%" % (n, dist.pmf(n)*100))


#print("Cumulative prob of each successful outcome:")
#for n in range(10, 110, 10):
#    print("Cumulative rob of %d successes: %.3f%%" % (n, dist.cdf(n)*100))p


#plot the PMF
x_values = list(range(1, 101, 1))
# print(x_values)
distplot = [binom.pmf(x,k,p) for x in x_values]
pyplot.bar(x_values, distplot)
pyplot.show()