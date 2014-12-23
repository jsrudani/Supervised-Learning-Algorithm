#!/usr/bin/python -tt
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: NaiveBayesClassificationFramework.py
#* Version: 1.0
#*
#***********************
__author__ = 'JigarRudani'
from collections import defaultdict

def naivebayes_classifier(trainingLst, testLst):

    _total_plus_one = 0
    _total_minus_one = 0
    _count_of_each_element = defaultdict()
    _count_element_for_plusone_label = defaultdict()
    _count_element_for_minusone_label = defaultdict()
    _conditinal_prob_for_minusone_label = defaultdict()
    _conditinal_prob_for_plusone_label = defaultdict()
    _attribute_train_dict = defaultdict()
    _attribute_test_dict = defaultdict()

    # Get the unique label from both the Training and Test data
    _attribute_train_dict, _attribute_test_dict = _get_unique_label_for_each_attribute(trainingLst, testLst)

    # Initialise all data structures to default value of 0 which includes all the attributes along with its unique values
    _count_of_each_element = _get_combine_list(_attribute_train_dict,_attribute_test_dict)
    _count_element_for_plusone_label = _get_combine_list(_attribute_train_dict,_attribute_test_dict)
    _count_element_for_minusone_label = _get_combine_list(_attribute_train_dict,_attribute_test_dict)

    _total_observation = len(trainingLst)

    # Traverse Training list to count the number of each unique Label
    for rows in trainingLst:
        if (rows[0] == -1):
            _total_minus_one += 1
        else:
            _total_plus_one += 1

    # Count the number of each unique label of each attribute for each label +1 and -1
    for rows in trainingLst:
        for element in range(1, len(trainingLst[0])):
            if (rows[element] != 0):
                index = _get_index_of_key_from_dict(_count_of_each_element[element],rows[element])
                if (index != -1):
                    _count_of_each_element[element][index][rows[element]] += 1
                if (rows[0] == -1):
                    index = _get_index_of_key_from_dict(_count_element_for_minusone_label[element], rows[element])
                    if (index != -1):
                        _count_element_for_minusone_label[element][index][rows[element]] += 1
                elif (rows[0] == 1):
                    index = _get_index_of_key_from_dict(_count_element_for_plusone_label[element],rows[element])
                    if (index != -1):
                        _count_element_for_plusone_label[element][index][rows[element]] += 1

    # Calculate Probablity of each label
    prob_minus_one = float(_total_minus_one)/float(_total_observation)
    prob_plus_one = float(_total_plus_one)/float(_total_observation)
    #print("Probablity of +1 and -1 is %f %f" % (prob_minus_one, prob_plus_one))

    _conditinal_prob_for_minusone_label[0] = prob_minus_one
    _conditinal_prob_for_plusone_label[0] = prob_plus_one

    # Calculate the probablity of each unique Label of each attribute
    for attribute_key, attribute_value in _count_of_each_element.items():
        isLaplacian_Negative  = _get_laplacian_flag(_count_element_for_minusone_label[attribute_key])
        isLaplacian_Positive = _get_laplacian_flag(_count_element_for_plusone_label[attribute_key])
        _conditinal_prob_for_minusone_label[attribute_key] = _get_probablity(_count_element_for_minusone_label[attribute_key], isLaplacian_Negative, _total_minus_one)
        _conditinal_prob_for_plusone_label[attribute_key] = _get_probablity(_count_element_for_plusone_label[attribute_key], isLaplacian_Positive, _total_plus_one)

    # Return the Conditional Probablity model for each Label
    return (_conditinal_prob_for_minusone_label, _conditinal_prob_for_plusone_label)

def predict_label(_attribute_list, _attribute_uniq_val_list, _conditinal_prob_for_minusone_label, _conditinal_prob_for_plusone_label):

    _predicted_attr_label_list = []

    # Predict the Label for _attribute_list data with model trained by Training data
    for attr_rows,uniq_rows in zip(_attribute_list,_attribute_uniq_val_list):
        _prob_result_minus_one = _conditinal_prob_for_minusone_label[0]
        _prob_result_plus_one = _conditinal_prob_for_plusone_label[0]
        for attr_element, uniq_element in zip(attr_rows, uniq_rows):
            index = _get_index_of_key_from_dict(_conditinal_prob_for_minusone_label[attr_element], uniq_element)
            if (index != -1):
                _prob_result_minus_one *= float(_conditinal_prob_for_minusone_label[attr_element][index][uniq_element])

            index = _get_index_of_key_from_dict(_conditinal_prob_for_plusone_label[attr_element], uniq_element)
            if (index != -1):
                _prob_result_plus_one *= float(_conditinal_prob_for_plusone_label[attr_element][index][uniq_element])

        if (_prob_result_plus_one > _prob_result_minus_one):
            _predicted_attr_label_list.append(1)
        else:
            _predicted_attr_label_list.append(-1)

    # Return predicted label for attribute list
    return _predicted_attr_label_list

def _get_unique_label_for_each_attribute(trainingLst, testLst):

    _attribute_train_dict = defaultdict()
    _attribute_test_dict = defaultdict()

    # Get the unique label from Test data and prepare dictionary --> attribute : [{uniq1:label},{uniq2:label}...,{uniqn:label}}]
    for attribute in range(1, len(testLst[0])):
        dummylist = []
        for rows in testLst:
            if ({rows[attribute]:rows[0]} not in dummylist):
                if (rows[0] == 1):
                    dummylist.append({rows[attribute]:rows[0]})
                elif (rows[0] == -1):
                    dummylist.append({rows[attribute]:rows[0]})
        _attribute_test_dict[attribute] = dummylist

    # Get the unique label from Training data and prepare dictionary --> attribute : [uniq1,uniq2...,uniqn]
    for attribute in range(1, len(trainingLst[0])):
        dummylist = []
        for rows in trainingLst:
            if (rows[attribute] not in dummylist):
                dummylist.append(rows[attribute])
        _attribute_train_dict[attribute] = dummylist

    # Return the structure prepared
    return (_attribute_train_dict, _attribute_test_dict)

def _get_combine_list(_attribute_train_dict,_attribute_test_dict):

    _combine_attribute_list_prepared = defaultdict()

    # Combine each attribute unique values from both the training and test data
    for train_key, train_val in _attribute_train_dict.items():
        dummy_list = []
        for test_key, test_val in _attribute_test_dict.items():
            if (test_key == train_key):
                for each_item in train_val:
                    for items in test_val:
                        for key in items.keys():
                            if (key == each_item):
                                if({key:0} not in dummy_list):
                                    dummy_list.append({key:0})
                            elif({each_item:0} not in dummy_list):
                                dummy_list.append({each_item:0})
                            elif({key:0} not in dummy_list):
                                dummy_list.append({key:0})
                            break
                _combine_attribute_list_prepared[train_key] = dummy_list
                break
    return _combine_attribute_list_prepared

def _get_index_of_key_from_dict(mapList, search_key):

    i = 0
    for items in mapList:
        for keys in items.keys():
            if(keys == search_key):
                return i
            else:
                i += 1
    return -1

def _get_laplacian_flag(attribute_value_list):

    # Length of list is 1 then check for key == 0
    if (len(attribute_value_list) == 1):
        if (0 in attribute_value_list[0].keys()):
            return True
        elif (0 in attribute_value_list[0].values()):
            return True
    else:
        for items in attribute_value_list:
            for key, values in items.items():
                if (values == 0 and key != 0):
                    return True
    return False

def _get_probablity(attribute_value_list, isLaplacian, _total_count):

    prob_list = []

    # Get the total count of unique label for each attribute
    if ({0: 0} in attribute_value_list):
        _total_count_uniq_label = len(attribute_value_list) - 1
    else:
        _total_count_uniq_label = len(attribute_value_list)

    # calculate the conditional probabality for each attribute uniq values
    for attribute in attribute_value_list:
        for key, value in attribute.items():
            if (key != 0):
                # Check if isLaplacian flag is True or not.
                # It True then increment the count of each attributes unique value and also increment count of dividing factor by count of uniq val
                if (isLaplacian):
                    prob = float(value + 1)/float(_total_count + _total_count_uniq_label)
                else:
                    prob = float(value)/float(_total_count)
                # Store the output in the list in the form of {uniq_val : prob_value}
                prob_list.append({key:prob})
    return prob_list