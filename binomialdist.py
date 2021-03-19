from scipy.stats import binom

p = 0.5
k= 100

#'mvsk': 'mean', 'variance'm 'skewness', 'kurtosis'
mean, var, _ , _ = binom.stats(k ,p, moments='mvsk')
print("Mean:  %.3f, variance: %.3f" % (mean, var))

## code-runne.run not found issue
