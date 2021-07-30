from pandas import DataFrame
from pandas import concat
import pandas as pd


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out:  Number of observations as output (y).
        dropnan:  Boolean; whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for Supervised Learning.    
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #forecast sequence (t, t+!, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i ==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    #chuck it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    #drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

ds = pd.read_csv('C:\Github\WS8\Data\Elec_daily_Prc_Dmd_update_file.csv')
print(ds)

raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
#values = raw.values
values = ds.values

data = series_to_supervised(values, 1, 2)
print(data)
