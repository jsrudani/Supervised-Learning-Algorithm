#!/usr/bin/python -tt
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: NaiveBayes.py
#* Version: 1.0
#*
#***********************
__author__ = 'JigarRudani'
import sys
import Utilities
import NaiveBayesClassificationFramework
import EvaluationReport

def main():

    # Getting the input from Command Line and feed to read_input()
    #print("Reading input from command line...")
    if len(sys.argv) <= 2:
        progName = sys.argv[0]
        print('\n Usage: %s [Training Data] [Test Data]' % (progName))
        sys.exit(-1)
    else:
        _training_data_file = sys.argv[1]
        _test_data_file = sys.argv[2]

        # Check if file exist or not
        is_Exist = Utilities.check_file_exist(_training_data_file,_test_data_file)
        if (is_Exist):
            #print("Processing Training and Test Data...")
            # Called the read_input() and returns training and test data list
            train_DataLst, training_Attributelst, training_Labellst, training_Attribute_Uniqval_lst, test_DataLst, test_Attributelst, test_Labellst, test_Attribute_Uniqval_lst = Utilities.read_input(_training_data_file,_test_data_file)
        else:
            sys.exit(-1)
        #print("Processing Completed")
        #print("Classifier running...")

        # Give training data to trained the model using Naive Bayes Classifier
        _conditional_prob_for_minusone_label, _conditional_prob_for_plusone_label = NaiveBayesClassificationFramework.naivebayes_classifier(train_DataLst, test_DataLst)
        #print("Classification completed")

        # Predict the label using Trained data
        predicted_training_label = NaiveBayesClassificationFramework.predict_label(training_Attributelst, training_Attribute_Uniqval_lst, _conditional_prob_for_minusone_label, _conditional_prob_for_plusone_label)
        predicted_test_label = NaiveBayesClassificationFramework.predict_label(test_Attributelst, test_Attribute_Uniqval_lst, _conditional_prob_for_minusone_label, _conditional_prob_for_plusone_label)

        #print("Generating Evaluation Report...")
        # Prepare Evaluation Report with the Actual and Predicted Label
        EvaluationReport.eval_report(training_Labellst, predicted_training_label)
        EvaluationReport.eval_report(test_Labellst, predicted_test_label)
        #print("Evaluation Report completed")

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
