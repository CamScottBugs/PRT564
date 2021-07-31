#import required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# %matplotlib inline   -- Jupyter notebook use

#read the data
df = pd.read_csv('data\Elec_daily_Prc_Dmd.csv', parse_dates=['date'])    #, 'Time']
dfQ = df.drop([ 'school_day','holiday'], axis = 1)

#check the dtypes
print(dfQ.dtypes)
dfQ['date'] = pd.to_datetime(dfQ.date , format = '%d/%m/%Y')
data = dfQ.drop(['date'], axis=1)
data.index = dfQ.date
print(data)


# #missing value treatment
# cols = data.columns
# for j in cols:
#     for i in range(0,len(data)):
#        if data[j][i] == -200:
#            data[j][i] = data[j][i-1]

# #checking stationarity
# from statsmodels.tsa.vector_ar.vecm import coint_johansen
# #since the test works for only 12 variables, I have randomly dropped
# #in the next iteration, I would drop another and check the eigenvalues
# johan_test_temp = data.drop([ 'min_temperature','max_temperature','ave_temp','school_day','rainfall'], axis=1)
# coint_johansen(johan_test_temp,-1,1).eig


#creating the train and validation set
X = data.values
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]
print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
plt.plot(train)
plt.plot([None for i in train] + [X for X in test]) 
plt.show()

# train = data[:int(0.8*(len(data)))]
# valid = data[int(0.8*(len(data))):]

#trainB = np.asarray(train).dtype
#trainC = pd.DataFrame(range(100)).dtypes

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

#est = sm.OLS(y, train.astype(float)).fit()

# model = VAR(endog=train)
# model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))


