#from load_csv import load_csv
import xgboost as xgb
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd

 
#boston = load_boston()
# d = df['data/Elec_daily_Prc_Dmd_Quant.csv']  #, index_col=[0], parse_dates=[0])
d = pd.read_csv('data/Elec_daily_Prc_Dmd_Quant.csv')
print(d)
x, y = d.data, d.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)



# #split TRAIN / TEST data on Vic Lockdown Date
# split_date = '23-Mar-2020'
# xtrain = d.loc[d.index <= split_date].copy()
# xtest = d.loc[d.index > split_date].copy()
# ytrain = d.loc[d.index <= split_date].copy()
# ytest = d.loc[d.index > split_date].copy()


# X_train, y_train = create_features(d_train, label='demand')
# X_test, y_test = create_features(d_test, label='demand')

xgbr = xgb.XGBRegressor(verbosity=0)
print(xgbr)

xgbr.fit(xtrain, ytrain)
 
score = xgbr.score(xtrain, ytrain)   
print("Training score: ", score) 
 
# - cross validataion 
scores = cross_val_score(xgbr, xtrain, ytrain, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
 
ypred = xgbr.predict(xtest)
mse = mean_squared_error(ytest, ypred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()