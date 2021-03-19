# Calculate arithmetic mean of a list of numbers
# Input: a list of numbers
# Output: a mean value

def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Test
# squares = list(map(lambda x: x**2, range(10)))  # a list of squares of integers 0, 1, ..., 9.
# print(squares)
# print("Mean: %.3f" % mean(squares))
