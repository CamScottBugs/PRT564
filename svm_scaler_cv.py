# Codes adapted from:
# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python


from sklearn import datasets, preprocessing, svm, metrics
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from statistics import mean

# A. Load dataset
dataset = datasets.load_breast_cancer()

# B. Prepate the dataset
X = dataset.data
y = dataset.target

# C. Prepare Cross Validator and the scoring schemes

scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

# D. Build a cross validation pipeline
pipe = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='linear'))


# E. Evaluate the model
scores = cross_validate(pipe, X, y, cv=10, scoring=scoring)


# accuracy
print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy'])*100))

# precision
print("Mean precision: %.3f " % (mean(scores['test_precision_macro'])))

# recall
print("Mean recall: %.3f" % (mean(scores['test_recall_macro'])))

# # F1 (F-Measure)
print("F1: %.3f" % (mean(scores['test_f1_macro'])))
