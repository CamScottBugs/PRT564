from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('data/Elec_daily_Prc_Dmd.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = None # specify 'None' if want to read whole file
fields = ['demand', 'RRP', 'solar_exposure']
# complete_dataset.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('data/Elec_daily_Prc_Dmd.csv', delimiter=',', nrows = nRowsRead, usecols = fields)  #, ncols = nColsRead)
df1.dataframeName = 'Elec_daily_Prc_Dmd.csv'
#nCols = df1.dataframeName[['demand', 'RRP', 'solar_exposure']]
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
#print(df1)


####==============================================================
# # Distribution graphs (histogram/bar graph) of column data
# def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
#     nunique = df.nunique()
#     df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
#     nRow, nCol = df.shape
#     columnNames = list(df)
#     nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
#     plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
#     for i in range(min(nCol, nGraphShown)):
#         plt.subplot(nGraphRow, nGraphPerRow, i + 1)
#         columnDf = df.iloc[:, i]
#         if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
#             valueCounts = columnDf.value_counts()
#             valueCounts.plot.bar()
#         else:
#             columnDf.hist()
#         plt.ylabel('counts')
#         plt.xticks(rotation = 90)
#         plt.title(f'{columnNames[i]} (column {i})')
#     plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
#     plt.show()
    

##==============================================
# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    #df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10] #10?
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.4, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr.coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter Density Plot', fontsize=20)
    plt.show()
    
print(df1.head(5))   #!! kill
#plotPerColumnDistribution(df1, 10, 5)
#plotCorrelationMatrix(df1, 8)

plotScatterMatrix(df1, 15, 17.5)

# df2 = pd.DataFrame('data/Elec_daily_Prc_Dmd_Quant.csv'), columns=['demand', 'RRP', 'solar_exposure']
# pd.plotting.scatter_matrix(df2, alpha=0.2)