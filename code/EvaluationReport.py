#!/usr/bin/python -tt
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: EvaluationReport.py
#* Version: 1.0
#*
#***********************
from math import pow
__author__ = 'JigarRudani'

def eval_report(training_Labellst, predicted_training_label):

    # Beta value for FBetaScore()
    beta_5 = 0.5
    beta_2 = 2

    # Calculate the TP, FP, TN, FN for Training and Test data
    #print("TP FN FP TN")

    # Training Data
    _True_Positive, _False_Positive, _True_Negative, _False_Negative = count_metrics(training_Labellst, predicted_training_label)
    _total_positive_count = _True_Positive + _False_Negative
    _total_negative_count = _True_Negative + _False_Positive

    accuracy = _get_Accuracy(_True_Positive, _True_Negative, _total_positive_count, _total_negative_count)
    error_rate = _get_Error_Rate(_False_Positive, _False_Negative, _total_positive_count, _total_negative_count)
    sensitivity = _get_Sensitivity(_True_Positive, _total_positive_count)
    specificity = _get_Specificity(_True_Negative, _total_negative_count)
    precision = _get_Precision(_True_Positive,_False_Positive)
    f1score = _get_F1Score(precision, sensitivity)
    fbeta5 = _get_FBetaScore(beta_5, precision, sensitivity)
    fbeta2 = _get_FBetaScore(beta_2, precision, sensitivity)

    # # Print all the metrics
    # print "Accuracy \t", accuracy
    # print "Error Rate \t", error_rate
    # print "Sensitivity \t", sensitivity
    # print "Specificity \t", specificity
    # print "Precision \t", precision
    # print "F1Score \t", f1score
    # print "Fbeta (0.5) \t", fbeta5
    # print "Fbeta (0.2)\t", fbeta2

    #print(accuracy, error_rate, sensitivity, specificity, precision, f1score, fbeta)

'''
accuracy, recognition rate accuracy, recognition rate TP + TN/P + N
error rate, misclassification rate FP + FN/P + N
sensitivity, true positive rate,recall TP/P
specificity, true negative rate TN/N
precision TP/TP + FP
F, F1, F-score, harmonic mean of precision and recall 2 * precision * recall/precision + recall
Fbeta , where beta is a non-negative real number (1 + beta^2) * precision * recall/beta^2 * precision + recall
'''
def _get_Accuracy(_True_Positive, _True_Negative, _total_positive_count, _total_negative_count):
    return (float(_True_Positive + _True_Negative)/float(_total_positive_count + _total_negative_count))

def _get_Error_Rate(_False_Positive, _False_Negative, _total_positive_count, _total_negative_count):
    return (float(_False_Positive + _False_Negative)/float(_total_positive_count + _total_negative_count))

def _get_Sensitivity(_True_Positive, _total_positive_count):
    return (float(_True_Positive)/float(_total_positive_count))

def _get_Specificity(_True_Negative, _total_negative_count):
    return (float(_True_Negative)/float(_total_negative_count))

def _get_Precision(_True_Positive,_False_Positive):
    return (float(_True_Positive)/float(_True_Positive + _False_Positive))

def _get_F1Score(precision, sensitivity):
    return (float((2 * precision * sensitivity))/float(precision + sensitivity))

def _get_FBetaScore(beta, precision, sensitivity):
    return (float((float(1 + pow(beta,2)) * float(precision) * float(sensitivity))/(float(pow(beta,2)) * float(precision) + float(sensitivity))))

def count_metrics(given_label, predicted_label):

    # Count TP, FP, TN, FN for given_label and predicted_label
    _TP_ = 0
    _FP_ = 0
    _TN_ = 0
    _FN_ = 0

    for gvnlbl, pred_lbl in zip(given_label, predicted_label):

        if (gvnlbl == 1 and pred_lbl == 1):
            _TP_ += 1
        elif (gvnlbl == -1 and pred_lbl == -1):
            _TN_ += 1
        elif (gvnlbl == 1 and pred_lbl == -1):
            _FN_ += 1
        elif (gvnlbl == -1 and pred_lbl == 1):
            _FP_ += 1

    print _TP_, _FN_, _FP_, _TN_

    # Return the calculated metrics
    return (_TP_, _FP_, _TN_, _FN_)
