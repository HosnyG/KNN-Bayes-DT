from basics import read_examples
p_yes = 0
p_no = 0


# implement naive bayes prediction given train & test file, return accuracy according to test examples
def naive_bayes(train_file, test_file):
    training, _ = read_examples(train_file)
    testing, _ = read_examples(test_file)
    global p_yes, p_no  # to calculate once
    p_yes = p_of_tag('yes', training)   # |examples with tag yes| / |examples|
    p_no = p_of_tag('no', training)
    optimization_dic = {}  # as cache, reduces run time from 3 minutes to 1 sec.
    # calculate  true positive and false negative
    right_prediction = sum(t.tag == classify_bayes(t.features_values, training, optimization_dic) for t in testing)
    return right_prediction / len(testing)  # Accuracy


# given tag (yes or no) , return it's proportion according to all examples
def p_of_tag(tag, examples):
    return sum(ex.tag == tag for ex in examples)/len(examples)


# calculate Pr(feature = value | tag )
def p_of_feature_value_given_tag(feature, value, tag, examples):
    return sum(ex.features_values[feature] == value and ex.tag == tag for ex in examples) / sum(
        ex.tag == tag for ex in examples)


# given example (without tag) , classify it (return yes or no)
# class(a1,...,an) = MAX{Pr(yes)*∏ Pr(aj = v | yes) , Pr(no)*∏ Pr(aj=v| no)}
def classify_bayes(attr_vector, examples, dic):
    yes_prediction = 1
    no_prediction = 1
    for i in range(len(attr_vector)):  # i is feature
        if (i, attr_vector[i], 'yes') in dic:  # calculated before
            yes_prediction *= dic[(i, attr_vector[i], 'yes')]
        else:
            conditional_p = p_of_feature_value_given_tag(i, attr_vector[i], 'yes', examples)  # Pr(ai=v |yes)
            yes_prediction *= conditional_p
            dic[(i, attr_vector[i], 'yes')] = conditional_p  # cache
        if (i, attr_vector[i], 'no') in dic:  # calculated before
            no_prediction *= dic[(i, attr_vector[i], 'no')]
        else:
            conditional_p = p_of_feature_value_given_tag(i, attr_vector[i], 'no', examples)  # Pr(ai=v |no)
            no_prediction *= conditional_p
            dic[(i, attr_vector[i], 'no')] = conditional_p
    if yes_prediction * p_yes >= no_prediction * p_no:
        return 'yes'
    return 'no'
