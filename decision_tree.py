from math import log
from basics import Node, Edge, read_examples
attr_list = []  # will keep attributes name in the given order
attr_dic = dict()  # will keep attr name as key , and the value is list of attr possible values


# implement dt prediction given train & test file return accuracy according to test examples and the tree
def decision_tree(train_file, test_file):
    global attr_list
    training, attr_list = read_examples(train_file)
    init_attributes_dic(training)
    testing, _ = read_examples(test_file)
    dt = ID3(training, list(attr_list), 'yes')  # tree
    # calculate  true positive and false negative
    right_prediction = sum(t.tag == classify_dt(t.features_values, dt) for t in testing)
    return right_prediction / len(testing), dt  # Accuracy


# initialize attributes values (sorted), according to the values appear in the file,
def init_attributes_dic(examples):
    for i in range(len(attr_list)):
        attr_dic[attr_list[i]] = sorted((list(set(ex.features_values[i] for ex in examples))))


# implementing ID3 algorithm, return the decision tree (as root node)
def ID3(examples, attributes, default):
    if len(examples) == 0:
        return Node(default)
    elif same_classification(examples):
        return Node(examples[0].tag)
    elif len(attributes) == 0:
        return Node(MODE(examples))
    else:
        best = choose_attribute(attributes, examples)
        tree = Node(best)
        for v in attr_dic[best]:
            examples_i = [ex for ex in examples if ex.features_values[attr_list.index(best)] == v]
            subtree = ID3(examples_i, [a for a in attributes if a != best], MODE(examples))
            tree.add_edge(Edge(v, subtree))
        return tree


# return true if all examples classified as yes or no ,false otherwise
def same_classification(examples):
    return len(set(ex.tag for ex in examples)) == 1


# given examples ,return the dominant class(tag)
def MODE(examples):
    if sum(ex.tag == 'yes' for ex in examples) > sum(ex.tag == 'no' for ex in examples):
        return 'yes'
    return 'no'


# given examples , return (num of examples classified as yes , num of examples classified as no)
def pos_neg(examples):
    return sum(ex.tag == 'yes' for ex in examples), sum(ex.tag == 'no' for ex in examples)


# given pair of (+,-) , calculate the entropy .
def Entropy(pn):
    pos = pn[0]/(pn[0]+pn[1])
    neg = pn[1]/(pn[0]+pn[1])
    if pos == 0 or neg == 0:
        return 0
    else:
        return -pos * log(pos, 2) - neg * log(neg, 2)


# given examples and attribute , return the information gain of choosing this attribute
def Gain(examples, attribute):
    s = 0
    for v in attr_dic[attribute]:
        SV = [ex for ex in examples if ex.features_values[attr_list.index(attribute)] == v]
        if len(SV) == 0:
            continue
        s += (len(SV)/len(examples)) * Entropy(pos_neg(SV))
    return Entropy(pos_neg(examples)) - s


# determine which attribute to choose
def choose_attribute(attributes, examples):
    max_gain = -1  # always gain >=0
    best_attribute = None  # redundant initialize
    for attribute in attributes:  # get the attribute with max IG
        g = Gain(examples, attribute)
        if g > max_gain:
            max_gain = g
            best_attribute = attribute
    return best_attribute


# given example(attributes vector) and decision tree, classify it (yes or no)
def classify_dt(att_vector, tree):
    current_subtree = tree
    while not current_subtree.is_leaf():  # tree traversal
        for e in current_subtree.edges:
            if e.label == att_vector[attr_list.index(current_subtree.attr_name)]:
                current_subtree = e.to
                break
    return current_subtree.attr_name  # yes or no


# print the tree to file according to the given format
def print_tree(root, file, tab_counter=0):
    for e in root.edges:
        if tab_counter == 0:
            file.write(root.attr_name + '=' + e.label)
        else:
            file.write(tab_counter * '\t' + '|' + root.attr_name + '=' + e.label)
        if e.to.is_leaf():
            file.write(':' + e.to.attr_name + '\n')
        else:
            file.write('\n')
            print_tree(e.to, file, tab_counter+1)
