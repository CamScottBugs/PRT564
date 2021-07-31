import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

d = pd.read_csv('data/Elec_daily_Prc_Dmd_Quant.csv', index_col=[0], parse_dates=[0])

#d.head()
#d.info()

d['rainfall'] = d.fillna(0)['rainfall']
d['solar_exposure'] = d.fillna(method='ffill')['solar_exposure']
d['solar_exposure'] = d.fillna(method='bfill')['solar_exposure']
#d.info()

#print(d['demand'].head())

d['demand'].head().shift(1)
#print(d['demand'].head().shift(1))
d['demand'].head(30).rolling(window = 30).mean()
print(d['demand'].head(30).rolling(window = 30).mean())


#create sliding window demand 'feature importance chart' 
def create_features(df, label=None):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    #df['weekofyear'] = df['date'].dt.weekofyear
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
           'dayofyear','dayofmonth']  #,'weekofyear']      #Series.dt.weekofyear deprecated
    
    for d in ('7', '15', '30'):
        for c in ('lag', 'mean', 'std', 'max', 'min'):
            cols.append(f'demand_{d}_days_{c}')
    
    X = df[cols]
    if label:
        y = df[label]
        return X, y
    return X


#split TRAIN / TEST data on Vic Lockdown Date
split_date = '10-Oct-2018'
d_train = d.loc[d.index <= split_date].copy()
d_test = d.loc[d.index > split_date].copy()


X_train, y_train = create_features(d_train, label='demand')
X_test, y_test = create_features(d_test, label='demand')

#print(X_test.dtypes)

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=True) 

##!!  x
_ = plot_importance(reg, height=0.9 ,max_num_features = 10)
#plt.plot(ft_imp)

d_test['pred'] = reg.predict(X_test)
d_all = pd.concat([d_test, d_train], sort=False)

_ = d_all[['demand','pred']].plot(figsize=(15, 5))
#plt.show()

f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = d_all[['pred','demand']].plot(ax=ax,
                                              style=['-','-'])
ax.set_xbound(lower='10-06-2019', upper='10-06-2020')
ax.set_ylim(60000, 160000)
plot = plt.suptitle("Oct'19-Oct2020 Forecast vs Actuals")
#plt.show()

mean_squared_error(y_true=d_test['demand'],
                   y_pred=d_test['pred'])

mean_absolute_error(y_true=d_test['demand'],
                   y_pred=d_test['pred'])

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_true=d_test['demand'],
                   y_pred=d_test['pred'])


from pandas import to_datetime
from fbprophet import Prophet

prophet_data_train = d_train.rename(columns={"date": "ds", "demand":"y"})[['ds', 'y']]
prophet_data_train.head()

f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = prophet_data_train.plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(60000, 190000)
plot = plt.suptitle('Timeseries')

pb = Prophet()
pb.fit(prophet_data_train)


future = list()
for i in range(1, 13):
    date = '2018-%02d' % i
    future.append([date])
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= to_datetime(future['ds'])

pb_insample_forecast = pb.predict(future)
pb_insample_forecast


pb.plot(pb_insample_forecast)


prophet_data_test = d_test.rename(columns={"date": "ds"})[['ds']]
pb_out_of_sample_forecast = pb.predict(prophet_data_test)
pb.plot(pb_out_of_sample_forecast)
plt.show()

mean_absolute_percentage_error(y_true=d_test['demand'],
                   y_pred=pb_out_of_sample_forecast[['yhat']])