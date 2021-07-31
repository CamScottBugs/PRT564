from gaussian_probability_density import calculate_probability
from summarize_by_class import summarize_by_class

# Calculate the probabilities of predicting each class for a given row
# Input: summary statistics of a dataset, row
# Output: a dict (key:class_value, value: probability score)

def calculate_class_probabilities(summaries, row):
    
    # find the total rows in the dataset
    total_rows = sum([summaries[label][0][2] for label in summaries])  
    
    # initialise a Python dictionary
    probabilities = dict()
    
    # compute the probability for each class
    # for each class, this loop solves P(class|data) = P(x1|class) x P(x2 |class) x ... x P(class)
    for class_value, class_summaries in summaries.items():
        
        # solves P(class)
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        
        # solves P(x1|class) x P(x2 |class) x ... x P(class)
        for i in range(len(class_summaries)):   # for a dataset with 2 columns, range is 0 to 1
            
            # get the mean, standard dev of column i from rows belonging to the same class 
            mean, stdev, count = class_summaries[i]
            
            # row[i] access the value of row in column [i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)    
    
    # return a dictionary of P(class|data)
    return probabilities


# predict the class for a given row
# i.e. for a given row, choose class with the largest P(class|data) value.
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# # Test calculating class probabilities
# dataset = [[3.393533211,2.331273381,0],
# [3.110073483,1.781539638,0],
# [1.343808831,3.368360954,0],
# [3.582294042,4.67917911,0],
# [2.280362439,2.866990263,0],
# [7.423436942,4.696522875,1],
# [5.745051997,3.533989803,1],
# [9.172168622,2.511101045,1],
# [7.792783481,3.424088941,1],
# [7.939820817,0.791637231,1]]
# summaries = summarize_by_class(dataset)
# probabilities = calculate_class_probabilities(summaries, dataset[0])
# print(probabilities)

# Running this example prints the probabilities calculated for each class. 
# We can see that the probability of the first row belonging to the 0 class (0.0503) 
#   is higher than the probability of it belonging to the 1 class (0.0001). 
# We would therefore correctly conclude that it belongs to the 0 class.