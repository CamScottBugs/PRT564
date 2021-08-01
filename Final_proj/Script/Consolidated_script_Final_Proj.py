from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd
import time
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Descriptive:

for dirname, _, filenames in os.walk('data/Elec_daily_Prc_Dmd.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = None
fields = ['demand', 'RRP', 'solar_exposure']
df1 = pd.read_csv('data/Elec_daily_Prc_Dmd.csv', delimiter=',', nrows = nRowsRead, usecols = fields) 
df1.dataframeName = 'Elec_daily_Prc_Dmd.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
#print(df1)

# Scatter /density plotting 
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) 
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 3: 
        columnNames = columnNames[:3] 
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.4, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr.coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter Density Plot', fontsize=20)
    plt.show()   
plotScatterMatrix(df1, 15, 17.5)

#Predictive, pull 
d = pd.read_csv('data/Elec_daily_Prc_Dmd.csv', index_col=[0], parse_dates=[0])

#Sliding Window wrangling ->  TS to supervised ML regression
d['demand'].head().shift(1)
d['demand'].head(30).rolling(window = 30).mean()
#print(d['demand'].head(30).rolling(window = 30).mean())

#prepare for 'Feature Importance' chart 
d['rainfall'] = d.fillna(0)['rainfall']
d['solar_exposure'] = d.fillna(method='ffill')['solar_exposure']
d['solar_exposure'] = d.fillna(method='bfill')['solar_exposure']

#Create Dmd feature importance chart
def create_features(df, label=None):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['demand_7_days_lag'] = df['demand'].shift(7)
    df['demand_15_days_lag'] = df['demand'].shift(15)
    df['demand_30_days_lag'] = df['demand'].shift(30)
    df['demand_7_days_mean'] = df['demand'].rolling(window = 7).mean()
    df['demand_15_days_mean'] = df['demand'].rolling(window = 15).mean()
    df['demand_30_days_mean'] = df['demand'].rolling(window = 30).mean()
    df['demand_7_days_std'] = df['demand'].rolling(window = 7).std()
    df['demand_15_days_std'] = df['demand'].rolling(window = 15).std()
    df['demand_30_days_std'] = df['demand'].rolling(window = 30).std()
    df['demand_7_days_max'] = df['demand'].rolling(window = 7).max()
    df['demand_15_days_max'] = df['demand'].rolling(window = 15).max()
    df['demand_30_days_max'] = df['demand'].rolling(window = 30).max()
    df['demand_7_days_min'] = df['demand'].rolling(window = 7).min()
    df['demand_15_days_min'] = df['demand'].rolling(window = 15).min()
    df['demand_30_days_min'] = df['demand'].rolling(window = 30).min()
    
    cols = ['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth'] 
    
    for d in ('7', '15', '30'):
        for c in ('lag', 'mean', 'std', 'max', 'min'):
            cols.append(f'demand_{d}_days_{c}')
    
    X = df[cols]
    if label:
        y = df[label]
        return X, y
    return X


#Split TRAIN / TEST data on Vic Lockdown Date
split_date = '06-Oct-2018'    # ~35% Test
d_train = d.loc[d.index <= split_date].copy()
d_test = d.loc[d.index > split_date].copy()

X_train, y_train = create_features(d_train, label='demand')
X_test, y_test = create_features(d_test, label='demand')

#"Extreme Gradient Boosting (XGBoost)" Supervised ML Regression Model 
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=True) 

_ = plot_importance(reg, height=0.9 ,max_num_features = 10)

d_test['pred'] = reg.predict(X_test)
d_all = pd.concat([d_test, d_train], sort=False)

_ = d_all[['demand','pred']].plot(figsize=(15, 5))

#XGB model plot 
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = d_all[['pred','demand']].plot(ax=ax,
                                              style=['-','-'])
ax.set_xbound(lower='10-06-2019', upper='10-06-2020')
ax.set_ylim(60000, 160000)
plot = plt.suptitle("Oct'19-Oct2020 Forecast vs Actuals")


RMSE = mean_squared_error(y_true=d_test['demand'],   #  sqrt(MSE)
                   y_pred=d_test['pred'])**0.5
print("\nDemand \nRoot Mean Square Error: %.2f" % RMSE,'\n')

def root_mean_squared_percentage_error(y_true, y_pred):  #Home grown calc of RMSPE given y_true and y_pred 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return ((np.mean((y_true - y_pred)**2 / y_true**2))**0.5) * 100

MSPE = root_mean_squared_percentage_error(y_true=d_test['demand'],
                   y_pred=d_test['pred'])
print("Root Mean Square Percentage Error\n(prediction/actual): %.2f"% MSPE,'%\n')


MAE = (mean_absolute_error(y_true=d_test['demand'],
                   y_pred=d_test['pred']))
print("Mean Absolute Error: %.2f" % MAE,'\n')

def mean_absolute_percentage_error(y_true, y_pred):   #Calculation for MAPE given y_true and y_pred
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(y_true=d_test['demand'],
                   y_pred=d_test['pred'])
print("Mean Absolute Percentage Error\n(prediction/actual): %.2f"% MAPE,'%\n')

plt.show()
