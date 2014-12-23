#!/usr/bin/python -t
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: NBAdaBoostEnsembleMethod.py
#* Version: 1.0
#*
#***********************
__author__ = 'JigarRudani'
import random
import NaiveBayesClassificationFramework
import math
from collections import defaultdict


def nb_Adaboost_Classifier(train_Datalst, training_Attributelst,  training_Labellst, test_DataLst, training_Attribute_Uniqval_lst, _total_rounds):

    predicted_training_label = []
    predicted_test_label = []
    training_data = []
    _correctly_classified_tuples_index = []
    _error_threshold_ = 0.5
    _rounds = 0
    _err_rate_for_model = 0.0
    _first_run = True

    # Error of each Model selected and also store the conditional probablities use for this model
    _error_rate_dict_model = defaultdict(int)
    _conditional_prob_plus_model = []
    _conditional_prob_minus_model = []

    # Weight associated with each row
    _weight_of_each_tuple = defaultdict(int)

    #print("Inside AdaBoost Classifier")

    # Calculate Initial weight
    _weight_of_each_tuple = _initial_weight(train_Datalst, _weight_of_each_tuple)
    #print(_weight_of_each_tuple)

    # Loop through K rounds as Constant value specified
    while (_rounds < _total_rounds):

        #print(_rounds)
        # Empty the training data and attribute list
        training_data = []
        training_attribute = []
        training_label = []
        training_attribute_uniq_val = []

        # Iterate through Training data to select random tuple
        for rows in range(1, (len(train_Datalst) + 1)):
            _row_number = _get_random_sample(_weight_of_each_tuple, _first_run, rows)
            if (_row_number >= len(train_Datalst)):
                _row_number = _row_number%len(train_Datalst)
            training_data.append(train_Datalst[_row_number])
            training_attribute.append(training_Attributelst[_row_number])
            training_attribute_uniq_val.append(training_Attribute_Uniqval_lst[_row_number])
            training_label.append(training_Labellst[_row_number])

        # Pass the generated random sample to Naive Bayes and get the predicted result
        _conditional_prob_for_minusone_label, _conditional_prob_for_plusone_label = NaiveBayesClassificationFramework.naivebayes_classifier(training_data, test_DataLst)
        predicted_training_label = NaiveBayesClassificationFramework.predict_label(training_attribute, training_attribute_uniq_val,  _conditional_prob_for_minusone_label, _conditional_prob_for_plusone_label)

        # Calculate Error rate
        _err_rate_for_model, _correctly_classified_tuples_index = _get_error_rate(training_label, predicted_training_label, _weight_of_each_tuple)

        # Check if error rate is > 0.5 or not
        if (_err_rate_for_model < _error_threshold_):

            # Store the error count for this model as it is < 0.5
            _error_rate_dict_model[_rounds] = _err_rate_for_model
            _conditional_prob_plus_model.append(_conditional_prob_for_plusone_label)
            _conditional_prob_minus_model.append(_conditional_prob_for_minusone_label)

            # Update the weight for the correctly classified tuple
            for correct_tuple in _correctly_classified_tuples_index:
                _weight_of_each_tuple[correct_tuple] = (_weight_of_each_tuple[correct_tuple] * (_error_rate_dict_model[_rounds]/(1 - _error_rate_dict_model[_rounds])))

            # Normalise weight of each tuple. so that sum of all weights is equal to 1
            _sum_of_weights = sum(_weight_of_each_tuple.values())
            for each_tuple in _weight_of_each_tuple.keys():
                _weight_of_each_tuple[each_tuple] = float(_weight_of_each_tuple[each_tuple]/_sum_of_weights)

            # Increment round as error rate is acceptable for this model
            _rounds += 1
        _first_run = False

    # Return Error for each Model to be used in prediction
    return (_error_rate_dict_model, _conditional_prob_plus_model, _conditional_prob_minus_model)

def nb_Adaboost_Predict(_error_rate_dict_model, _conditional_prob_plus_model, _conditional_prob_minus_model, _attribute_data, training_Attribute_Uniqval_lst, _total_rounds):

    _predict_label = []

    # Iterate through Test data to predict the label for each tuple
    for attr_tuple, uniq_tuple in zip(_attribute_data,training_Attribute_Uniqval_lst):
        _weight_minus_model = [0]
        _weight_plus_model = [0]
        for rounds in range(0, _total_rounds):
            weight_alpha = float(math.log((float(1 - _error_rate_dict_model[rounds])/float(_error_rate_dict_model[rounds]))))
            _predicted_test_label_list = NaiveBayesClassificationFramework.predict_label([attr_tuple], [uniq_tuple], _conditional_prob_minus_model[rounds],_conditional_prob_plus_model[rounds])

            # Store the weight_alpha to corresponding class label
            if (_predicted_test_label_list[0] == 1):
                _weight_plus_model.append(weight_alpha)
            else:
                _weight_minus_model.append(weight_alpha)

        # Predict the label with maximum class weight
        if (sum(_weight_plus_model) > sum(_weight_minus_model)):
            _predict_label.append(1)
        else :
            _predict_label.append(-1)

    # Return the Prediction List
    return (_predict_label)

def _get_error_rate(given_label, predicted_label, _weight_of_each_tuple):

    _correctly_classified_tuples_index = []
    _index = 0
    _err_xj = 0
    _result_value = 0.0

    for gvnlbl, pred_lbl in zip(given_label, predicted_label):
        if (gvnlbl == 1 and pred_lbl == 1):
            _err_xj = 0
            _correctly_classified_tuples_index.append(_index)
            _index += 1
        elif (gvnlbl == -1 and pred_lbl == -1):
            _err_xj = 0
            _correctly_classified_tuples_index.append(_index)
            _index += 1
        elif (gvnlbl == 1 and pred_lbl == -1):
            _index += 1
            _err_xj = 1
        elif (gvnlbl == -1 and pred_lbl == 1):
            _index += 1
            _err_xj = 1
        _result_value += float(_weight_of_each_tuple[_index]) * float(_err_xj)

    # Return error value for this model and also correctly classified tuple index
    return (_result_value, _correctly_classified_tuples_index)

def _get_random_sample(_weight_of_each_tuple, _first_run, rows):

    # Generate random number to select random row
    _random_row = random.randint(0, len(_weight_of_each_tuple) - 1)
    _random_num = random.random()
    _sum_weight = _weight_of_each_tuple[0]
    _index = 0

    while (_random_num > _sum_weight):
        _sum_weight += _weight_of_each_tuple[_index]
        _index += 1
    return _index

def _initial_weight(train_Datalst, _weight_of_each_tuple):

    _total_length = len(train_Datalst)
    for rows in range(_total_length):
        _weight_of_each_tuple[rows] = float(1.0/_total_length)

    # Return the weight map which will be used in subsequent phase
    return _weight_of_each_tuple