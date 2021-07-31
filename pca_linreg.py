# This example demonstrates that applying PCA
# prior to running a Linear Regression does not necessarily lead to a better performance.
# Therefore, use PCA with care.


# evaluate model on the raw dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np 


# load the dataset
df = read_csv("data/housing.csv", header=None)

# retrieve DataFrame's content as a matrix
data = df.values

# split into input variables (X) and output variable (y)
X, y = data[:, :-1], data[:, -1]

# compare the cumulative explained variance versus number of PCA components
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# based on the above plot, we decide that 4 principal components could capture nearly
# 100% of cumulative variance in the data.
# use PCA to reduce the dimensionality of the original dataset
pca = PCA(n_components=4)
X_projected = pca.fit_transform(X)

# compare the shapes of original data vs. data with reduced-dimension
print(X.shape)
print(X_projected.shape)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_projected, y, test_size=0.33, random_state=1)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model
y_hat = model.predict(X_test)

# evaluate predictions
mae = mean_absolute_error(y_test, y_hat)
print('MAE: %.3f' % mae)