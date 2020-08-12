from basics import read_examples
from sys import maxsize


# implement knn prediction given train & test file and K, return accuracy according to test examples
def knn(train_file, test_file, k=5):
    training, _ = read_examples(train_file)
    testing, _ = read_examples(test_file)
    # calculate  true positive and false negative
    right_prediction = sum(t.tag == classify_knn(t.features_values, training, k) for t in testing)
    return right_prediction / len(testing)  # Accuracy


# compute hamming_distance, attributes vector as string
def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


# given example (attributes vector) , classify it according to knn algorithm
def classify_knn(attr_vector, examples, k):
    # get distances of all examples according to the input
    distances = [hamming_distance("".join(attr_vector), "".join(ex.features_values)) for ex in examples]
    k_nearest = get_k_nearest(distances, examples, k)  # get k nearest examples to the input
    # classify it according to the majority
    if sum(ex.tag == 'yes' for ex in k_nearest) > sum(ex.tag == 'no' for ex in k_nearest):
        return 'yes'
    return 'no'


# given list of distances , return k nearest examples
def get_k_nearest(distances, examples, k):
    k_nearest = []  # initiliaze
    for i in range(k):  # choose k nearest examples
        index = distances.index(min(distances))  # get nearest example index
        k_nearest.append(examples[index])  # add example itself to list
        distances[index] = maxsize  # changes it's distance to ∞
    return k_nearest
