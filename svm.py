# Codes adapted from:
# https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# A. Load dataset
dataset = datasets.load_breast_cancer()


# B. Explore the dataset
# print the names of all features (input variables)
print("Features: ", dataset.feature_names)

# print the classes in the output variable
print("Labels: ", dataset.target_names)

# print the shape of the dataset (number of samples, number of features)
print(dataset.data.shape)

# print the top-5 records in the dataset
print(dataset.data[0:5])

# print all labels in the dataset
print(dataset.target)


# C. Prepare dataset for training
# splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=109)



# D. Build an SVM model
# create an instance of the SVM classifier with a Linear kernel
model = svm.SVC(kernel='linear')

# train the model using the training set
model.fit(X_train, y_train)

# predict the classes in the test set
y_pred = model.predict(X_test)


# E. Evaluate the model
# accuracy
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))

# precision
print("Precision: %.3f " % metrics.precision_score(y_test, y_pred))

# recall
print("Recall: %.3f" % metrics.recall_score(y_test, y_pred))

# F1 (F-Measure)
print("F1: %.3f" % metrics.f1_score(y_test, y_pred))
