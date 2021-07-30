import numpy as np
import pandas as pd
import matplotlib as plt

#fixbug! %matplotlib inline  #renders alignment of printed outputs within Jupyter notebook

# from numpy import array, diag

df=pd.read_csv('data/Elec_daily_Prc_Dmd_update_file.csv')
df.head()

# update header
#df.columns=["date","demand"]
df.head()
df.describe()
df.set_index('date', inplace=True)

from pylab import rcParams
rcParams['figure.figsize']=15,7
df.plot()

# print(df)

from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df['demand'])  