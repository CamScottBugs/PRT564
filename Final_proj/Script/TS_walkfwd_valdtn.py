from pandas import read_csv
from matplotlib import pyplot

series = read_csv('data\Elec_daily_Dmd_2D.csv', header=0, index_col=0)
X = series.values
n_train = 500
n_records = len(X)
for i in range(n_train, n_records):
    train, test = X[0:i], X[i:i+1]
    print('train=%d, test=%d' % (len(train), len(test)))
                      