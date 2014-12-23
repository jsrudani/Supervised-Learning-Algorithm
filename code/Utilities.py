#!/usr/bin/python -tt
#**********************
#* Author: Jigar S. Rudani
#* Progam Name: Utilities.py
#* Version: 1.0
#*
#***********************
__author__ = 'JigarRudani'
import os
import re
import sys
from collections import defaultdict

def read_input(train_DataFile, test_DataFile):

    # Training and Test Data List
    test_DataLst = []
    tmptrain_DataLst = []
    test_Labellst = []

    try:
        _max_attribute_index = _get_max_length(train_DataFile, test_DataFile)

        # Reading Training Data
        with open (train_DataFile,'r') as train_file_descriptor:
            inputLine = train_file_descriptor.read().splitlines()
            for rows in inputLine:
                tmptrain_DataLst.append(rows)
            train_DataLst, training_Attributelst, training_Labellst, training_Attribute_Uniqval_lst = process_inputLine(tmptrain_DataLst, _max_attribute_index)

        tmptrain_DataLst = []

        # Reading Test Data
        with open (test_DataFile,'r') as test_file_descriptor:
            inputLine = test_file_descriptor.read().splitlines()
            for row in inputLine:
                tmptrain_DataLst.append(row)
            test_DataLst, test_Attributelst, test_Labellst, test_Attribute_Uniqval_lst= process_inputLine(tmptrain_DataLst, _max_attribute_index)

    except Exception as e:
        print("Something went wrong in reading file...", e)
        sys.exit(-1)

    # Return final training and test data list
    return (train_DataLst, training_Attributelst, training_Labellst,training_Attribute_Uniqval_lst, test_DataLst, test_Attributelst, test_Labellst, test_Attribute_Uniqval_lst)

def _get_max_length(train_DataFile, test_DataFile):

    _max_attribute_index = 0
    try:
        # Reading Training Data to find MAX Attribute Index
        with open (train_DataFile,'r') as train_file_descriptor:
            inputLine = train_file_descriptor.read().splitlines()
            for rows in inputLine:
                element = rows.split(' ')
                for item in element:
                    if (re.match('\d+[:]\d+',item)):
                        _temp_attribute_index = int(item.split(':')[0])
                        if (_temp_attribute_index > _max_attribute_index):
                            _max_attribute_index = _temp_attribute_index

        _temp_attribute_index = 0

        # Reading Test Data to find MAX Attribute Index
        with open (test_DataFile,'r') as test_file_descriptor:
            inputLine = test_file_descriptor.read().splitlines()
            for row in inputLine:
                element = row.split(' ')
                for item in element:
                    if (re.match('\d+[:]\d+',item)):
                        _temp_attribute_index = int(item.split(':')[0])
                        if (_temp_attribute_index > _max_attribute_index):
                            _max_attribute_index = _temp_attribute_index
    except Exception as e:
        print("Something went wrong in finding Max index...", e)
        sys.exit(-1)
    #print("Maximum Index in Training and Test file is %d" % (_max_attribute_index))

    # Return the Max index from both Training and Test Data
    return(_max_attribute_index)

def process_inputLine(rowLst, _max_attribute_index):

    training_list = []
    training_label_list = []
    training_attribute_list = []
    training_attribute_unique_value = []

    for row in rowLst:
        element = row.split(' ')
        tmptraining_attribute_list = []
        tmptraining_attribute_unique_value_list = []
        for item in element:
            if (re.match('\d+[:]\d+',item)):
                tmptraining_attribute_list.append(int(item.split(':')[0]))
                tmptraining_attribute_unique_value_list.append(int(item.split(':')[1]))
            else:
                training_label_list.append(int(item))
        training_attribute_list.append(tmptraining_attribute_list)
        training_attribute_unique_value.append(tmptraining_attribute_unique_value_list)

    # Prepare Attribute Column along with Attribute value. If attribute is not there then value 0 is inserted.
    for row in rowLst:
        element = row.split(' ')
        _attribute_present_dict = defaultdict(int)
        temp_training_list = []
        for item in element:
            if (re.match('\d+[:]\d+',item)):
                _temp_attribute_index = int(item.split(':')[0])
                _attribute_present_dict[_temp_attribute_index] = int(item.split(':')[1])
            else:
                temp_training_list.append(int(item))
        for i in range(1,_max_attribute_index + 1):
            if (_attribute_present_dict[i]):
                temp_training_list.append(_attribute_present_dict[i])
            else:
                temp_training_list.append(0)
        training_list.append(temp_training_list)

    #Print Training list
    #for items in training_list:
    #    print(items)

    #return Training Data after parsing input lines
    return (training_list, training_attribute_list, training_label_list, training_attribute_unique_value)

def check_file_exist(train_DataFile, test_DataFile):
    #print("Checking Existing...")
    #print("Training Data: ",train_DataFile)
    #print("Test Data: ", test_DataFile)
    if (not os.path.isfile(train_DataFile)):
        print('Error %s file not found' % train_DataFile)
        is_Exist = False
    else:
        is_Exist = True

    if (not os.path.isfile(test_DataFile)):
        print('Error %s file not found' % test_DataFile)
        is_Exist = False
    else:
        is_Exist = True

    # Returning is_Exist status
    if(is_Exist):
        #print("File Exist")
        return is_Exist
    else:
        print("Please specify Correct Path")
        return is_Exist