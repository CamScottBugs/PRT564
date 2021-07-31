# Codes adapted from:
# https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0 


from sklearn import datasets, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

import pandas as pd 


# A. Load dataset
dataset = datasets.load_breast_cancer()

# B. Prepate the dataset
X = dataset.data
y = dataset.target

# C. Prepare Cross Validator and the scoring schemes
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']


# D. Build models to compare
models = [
    ('SVM', svm.SVC(kernel='linear')),
    ('GaussianNB', GaussianNB())
]


# E. Evaluate the models

# prepare a list of DataFrames to store model's results
dfs = []

for name, model in models:
    
    # get a model's performance score
    scores = cross_validate(model, X, y, cv=10, scoring=scoring)
        
    # store the model's name and results as DataFrame
    this_df = pd.DataFrame(scores)
    this_df['model'] = name
    
    # store the current DataFrame into a list of DataFrames 
    dfs.append(this_df)


# combine all DataFrames into a single DataFrame
df_final = pd.concat(dfs, ignore_index=True)


# F. Print results
# accuracy
print("Mean accuracy comparison:")
print(df_final.groupby('model').mean()['test_accuracy'].astype(float).map("{:.3%}".format))
print("--------------------------")

# precision
print("Mean precision comparison:")
print(df_final.groupby('model').mean()['test_precision_macro'].astype(float).map("{:.3f}".format))
print("--------------------------")

# recall
print("Mean recall comparisons:")
print(df_final.groupby('model').mean()['test_recall_macro'].astype(float).map("{:.3f}".format))
print("--------------------------")

# F1 (F-Measure)
print("Mean F1 comparison:")
print(df_final.groupby('model').mean()['test_f1_macro'].astype(float).map("{:.3f}".format))
print("--------------------------")


