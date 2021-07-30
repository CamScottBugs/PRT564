import pandas as pd
import numpy as np
import datetime

#date wrangling package
from datetime import datetime
# import scikits.timeseries

dti = pd.to_datetime(
    ["1/1/2018", np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)]
    )

dti

dataset = pd.read_csv('C:\Github\WS8\Data\Elec_daily_Prc_Dmd_update_file.csv')
print(dataset)

# Format to datetime
#dataset['Datetime'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in dataset['Datetime']]


    

# # retrieve DataFrame's content as a matrix
# data = df.to_numpy()
# # split into input and output elements
# X, y = data[:, :-1], data[:, -1]   
# # summarize the shape of the dataset
# print(X.shape, y.shape)
# #print meta info
# print(df.info())
# #print stats
# print(df.describe())