from summarize_by_class import summarize_by_class
from calculate_class_probabilities import predict

# This is the complete form of our Naive Bayes algorithm.
def naive_bayes(train, test):
    
    # learn the distribution of data in the training set
    summarize = summarize_by_class(train)
    
    # initialise an empty list to store prediction results
    predictions = list()
    
    # from what the model learns from the training set, 
    # computes the probability of each class in the testing set
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
        
    return(predictions)
