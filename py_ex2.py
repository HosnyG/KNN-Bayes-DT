from decision_tree import decision_tree, print_tree
from KNN import knn
from Bayes import naive_bayes


# implementing the 3 classifiers : dt , naive bayes and knn, print the output to 'output.txt' file
def main():
    train_file = 'train.txt'
    test_file = 'test.txt'
    bayes_accuracy = naive_bayes(train_file, test_file)
    knn_accuracy = knn(train_file, test_file, k=5)
    dt_accuracy, tree = decision_tree(train_file, test_file)
    with open('output.txt', 'w') as f:
        print_tree(tree, f)
        f.write('\n{}\t{}\t{}\n'.format(round(dt_accuracy, 2), round(knn_accuracy, 2), round(bayes_accuracy, 2)))


if __name__ == "__main__":
    main()
