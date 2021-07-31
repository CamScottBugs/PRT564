from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt 


# A. Load dataset
dataset = datasets.load_breast_cancer()


# B. Prepare dataset for training
# splitting data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=109)


# C. Build an SVM model
# create an instance of the SVM classifier with a Linear kernel
model = svm.SVC(kernel='linear')

# train the model using the training set
model.fit(X_train, y_train)

# predict the classes in the test set
y_pred = model.predict(X_test)


# D. Evaluate the model
# accuracy
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))

# precision
print("Precision: %.3f " % metrics.precision_score(y_test, y_pred))

# recall
print("Recall: %.3f" % metrics.recall_score(y_test, y_pred))

# F1 (F-Measure)
print("F1: %.3f" % metrics.f1_score(y_test, y_pred))


# compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# pretty print Confusion Matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
disp.plot()
plt.show()
