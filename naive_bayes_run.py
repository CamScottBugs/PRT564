# Apply Naive Bayes to classify Iris flowers

from load_csv import load_csv
from str_column_conversions import str_column_to_float, str_column_to_int
from evaluation import evaluate_algorithm
from naive_bayes_algorithm import naive_bayes

# load dataset
filename = 'data/iris.csv'
dataset = load_csv(filename)

# convert values in 1st to 4th columns to floating point numbers
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)

# convert class labels to integers
str_column_to_int(dataset, len(dataset[0])-1)   

# set the validation fold
n_folds = 5

# run Naive Bayes on the dataset and evaluate its predictive performance
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
