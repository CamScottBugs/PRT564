# Codes adapted from:
# https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection 


from sklearn import datasets, svm, preprocessing
from sklearn.model_selection import train_test_split

# A. Load dataset
dataset = datasets.load_breast_cancer()

# B. Prepare the dataset
X = dataset.data
y = dataset.target

# splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=109)


# C. Apply data transformation to the training set and test set
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


# D. Build SVM models
# create two SVM models: one with scaler and another without scaler.
model = svm.SVC(kernel='linear').fit(X_train, y_train)
model_scaled = svm.SVC(kernel='linear').fit(X_train_transformed, y_train)


# E. Evaluate the models by accuracy
accuracy = model.score(X_test, y_test)
accuracy_scaled = model_scaled.score(X_test_transformed, y_test)

print("Accuracy comparison:")
print("model --> %.3f%%" % (accuracy * 100))
print("model_scaled --> %.3f%%" % (accuracy_scaled * 100))
