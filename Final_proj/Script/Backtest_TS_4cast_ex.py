from pandas import read_csv
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot

#load data
series = read_csv('data\Elec_daily_Dmd_2D.csv', header=0, index_col=0)

X = series.values
splits = TimeSeriesSplit(n_splits=3)
pyplot.figure(1)
index = 1
for train_index, test_index in splits.split(X):
    train = X[train_index]
    test = X[test_index]
    #train_size = int(len(X)*0.66)
    #train, test = X[0:train_size], X[train_size:len(X)]
    print('Observations: %d' % (len(train) +len(test)))
    print('Training Observations: %d' % (len(train)))
    print('Testing Observations: %d' % (len(test)))
    pyplot.subplot(310 + index)
    pyplot.plot(train)
    pyplot.plot([None for i in train] + [X for X in test])
    pyplot(xlabel('date'))
    index += 1
pyplot.show()


# #print(series.head())
# # series.plot()
# pyplot.plot(train)
# pyplot.plot([None for i in train] + [X for X in test])
# pyplot.show()

# TimeSeriesSplit - Training/Test observations:
# training_size = i * n_samples / (n_splits +1) + n_samples % (n_splits +1)
# test_size = n_samples / (n_splits +1)

