from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd

for dirname, _, filenames in os.walk('data/Elec_daily_Prc_Dmd.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = None
fields = ['demand', 'RRP', 'solar_exposure']
df1 = pd.read_csv('data/Elec_daily_Prc_Dmd.csv', delimiter=',', nrows = nRowsRead, usecols = fields) 
df1.dataframeName = 'Elec_daily_Prc_Dmd.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
print(df1)

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
