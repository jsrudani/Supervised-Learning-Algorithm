#!/usr/bin/python -tt
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: NBAdaBoost.py
#* Version: 1.0
#*
#***********************
_author_ = "JigarRudani"
import sys
import NBAdaBoostEnsembleMethod
import Utilities
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
        _total_rounds = 5

        # Check if file exist or not
        is_Exist = Utilities.check_file_exist(_training_data_file,_test_data_file)
        if (is_Exist):
            #print("Processing Training and Test Data...")

            # Called the read_input() and returns training and test data list
            train_DataLst, training_Attributelst, training_Labellst, training_Attribute_Uniqval_lst, test_DataLst, test_Attributelst, test_Labellst, test_Attribute_Uniqval_lst = Utilities.read_input(_training_data_file,_test_data_file)

            #print("Classifier running...")

            # Give training data to Adaboost Classifier which uses Naive Bayes as classification model
            _error_rate_dict_model,_conditional_prob_plus_model, _conditional_prob_minus_model = NBAdaBoostEnsembleMethod.nb_Adaboost_Classifier(train_DataLst, training_Attributelst, training_Labellst, test_DataLst, training_Attribute_Uniqval_lst, _total_rounds)

            _predicted_training_label_list = NBAdaBoostEnsembleMethod.nb_Adaboost_Predict(_error_rate_dict_model, _conditional_prob_plus_model, _conditional_prob_minus_model, training_Attributelst, training_Attribute_Uniqval_lst, _total_rounds)
            #print("Classification completed")

            #print("Predicting...")

            # Give Error rate for each model calculated by AdaBoost to predict the class label for test data
            _predicted_test_label_list = NBAdaBoostEnsembleMethod.nb_Adaboost_Predict(_error_rate_dict_model, _conditional_prob_plus_model, _conditional_prob_minus_model, test_Attributelst, test_Attribute_Uniqval_lst, _total_rounds)
            #print("Prediction completed")

            #print("Generating Evaluation Report...")
            EvaluationReport.eval_report(training_Labellst, _predicted_training_label_list)
            EvaluationReport.eval_report(test_Labellst, _predicted_test_label_list)

        else:
            sys.exit(-1)
        #print("Processing Completed")

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()
