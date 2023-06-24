#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../../data_api')
sys.path.append('../../../data')

import pandas as pd
from data_api import DataApi
from random import shuffle


# CLASS


'''
    This class handles doing 10-fold cross validation. It uses a list of indexes to partition 
    and shuffle the data into 10 partitions of roughly equal size. The class has one method to 
    get a training set given a data frame and a test set number, and another method to get a 
    test set given a data frame and a test set number.
'''


class CrossValidation:


    def __init__(self):
        self.DEBUG  = False

    
    # return shuffled list of indexes 1-10 used to put in data set row into a partition for cross validation
    def get_indexes_list(self, data_frame):
        indexes_list = []
        # calculate partition size
        slice_count = ((data_frame.shape[0]) - (data_frame.shape[0] % 10)) / 10 
        slice_spot = 1
        counter = 0
        for i in range (0, data_frame.shape[0]): # ensure we don't have any boundary problems
            counter += 1 
            indexes_list.append(slice_spot)
            if counter == slice_count and slice_spot == 10:
                pass
            elif counter == slice_count:
                slice_spot += 1 # need to double check how many rows are in
                counter = 0

        # randomly shuffle list of indexes so rows will randomly get assigned to partitions
        shuffle(indexes_list)
        # return list of randomly shuffled indexes
        return indexes_list


    # return training set given a data frame, test set number, and indexes list
    def get_training_set(self, data_frame, test_set_number, indexes_list):
        data_copy = data_frame.copy()
        test_set_list = []
        for i in range(data_frame.shape[0]):
            if indexes_list[i] == test_set_number:
                test_set_list.append(i)
            else:
                pass
        for drop_column in test_set_list:
            try:
                # pandas will throw error if column does not exist
                data_copy.drop(drop_column, inplace=True)
            except:
                pass
        # return training set data frame
        return data_copy

            
    # return test set given a data frame, test set number, and indexes list
    def get_test_set(self, data_frame, test_set_number, indexes_list):
        data_copy = data_frame.copy()
        test_set_list = []
        for i in range(data_frame.shape[0]):
            if indexes_list[i] == test_set_number:
                pass
            else:
                test_set_list.append(i)
        for drop_column in test_set_list:
            try:
                # pandas will throw error if column does not exist
                data_copy.drop(drop_column, inplace=True)
            except:
                pass
        # return test set data frame
        return data_copy

    