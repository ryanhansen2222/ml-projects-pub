#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi
import math
import random
import pandas as pd


# CLASS

'''
    This class handles preprocessing all the data sets. It contains methods for preprocessing 
    each specific data set according to what needs to be done for the given data set.
'''


class Preprocess:

    def __init__(self):
    	self.DEBUG = False
    	self.data_api_impl = DataApi('../../data/')


    # preprocess breast cancer data given raw breast cancer data
    def preprocess_breast_cancer_data(self, raw_breast_cancer_data):
        preprocessed_data_frame = raw_breast_cancer_data.copy().drop('ID NUMBER', axis=1)
        for row_index in range(raw_breast_cancer_data.shape[0]):
            row_data = raw_breast_cancer_data.iloc[row_index, :]
            if '?' in row_data.values:
                preprocessed_data_frame.drop(row_data.name, inplace=True)
        return preprocessed_data_frame


    # preprocess glass data given raw glass data
    def preprocess_glass_data(self, raw_glass_data):
        minimum = [1.5112, 10.73, 0, 0.29, 69.81, 0, 5.43, 0, 0]
        maximum = [1.5339, 17.38, 4.49, 3.5, 75.41, 6.21, 16.19, 3.15, .51]
        return self.discrete(raw_glass_data.drop('ID NUMBER', axis=1), minimum, maximum)


    # preprocess iris data given raw iris data
    def preprocess_iris_data(self, raw_iris_data):
        minimum = [4.3, 2.0, 1.0, .1]
        maximum = [7.9, 4.4, 6.9, 2.5]
        return self.discrete(raw_iris_data, minimum, maximum)


    # bin raw data by putting every column value into one of two bins: less than avg (or) greater than avg
    def preprocess_data_avg_bin(self, raw_data):
        preprocessed_data = raw_data.copy()
        col_avg_vals = [] # list of average values for each column

        for column_label, column_vals in preprocessed_data.items():

            if column_label != 'CLASS':
                # get average value of column
                col_avg = preprocessed_data[column_label].mean()

                for row_index in range(preprocessed_data.shape[0]):
                    curr_val = preprocessed_data.loc[row_index, column_label]
                    binned_val = 1 # bin value = 1 if value is less than avg
                    if curr_val > col_avg:
                        # bin value = 2 if value is greater than avg
                        binned_val = 2

                    # replace data set value with binned value (1 or 2)
                    preprocessed_data.loc[row_index, column_label] = binned_val

        # return average-binned data set
        return preprocessed_data


    # preprocess house votes data given raw house votes data
    def preprocess_house_votes_data(self, raw_house_votes_data):
        new_frame = raw_house_votes_data.copy()
        # replace missing values with random yes/no value
        for label, content in new_frame.items():
            y = 0
            for m in content:
                if new_frame[label][y] == '?':
                    filler = random.randint(0,1)
                    marker = 'yes'
                    if filler == 1:
                        marker = 'y'
                    else:
                        marker = 'n'
                    new_frame[label][y] = marker
                y = y+1
        # return preprocessed data set with missing values replaced
        return new_frame


    # main method for this class: preprocess a given data frame according to what needs to be done for that specific data set
    def preprocess_raw_data_frame(self, raw_data_frame, raw_data_frame_name):
    	preprocessed_data_frame = raw_data_frame.copy()
    	if raw_data_frame_name == 'breast_cancer':
    		preprocessed_data_frame = self.preprocess_breast_cancer_data(raw_data_frame)
    	elif raw_data_frame_name == 'glass':
    		preprocessed_data_frame = self.preprocess_data_avg_bin(raw_data_frame.drop('ID NUMBER', axis=1))
    	elif raw_data_frame_name == 'iris':
    		preprocessed_data_frame = self.preprocess_data_avg_bin(raw_data_frame)
    	elif raw_data_frame_name == 'soybean_small':
    		preprocessed_data_frame = raw_data_frame # soybean_small data does not need any preprocessing
    	elif raw_data_frame_name == 'house_votes':
    		preprocessed_data_frame = self.preprocess_house_votes_data(raw_data_frame)
    	else:
    		print('ERROR: Unknown raw_data_frame_name => ' + raw_data_frame_name)

    	return preprocessed_data_frame


    # DISCRETIZATION
    	

    # discretize the data given a set number of bins
    def discrete(self, data_frame, minimum, maximum):
        new_frame = data_frame.copy()
        bin_index = 0
        num_bins = 2

        for column_label, column_values in new_frame.items():

            if column_label != 'CLASS':
                binsize = (maximum[bin_index] - minimum[bin_index]) / num_bins
                row_num = 0

                for col_val in column_values:
                    x = (col_val - minimum[bin_index]) / binsize
                    bin_value = int(math.ceil(x))
                    # enforce lower and upper bounds
                    if bin_value <= 0:
                        bin_value = 1
                    if bin_value > num_bins:
                        bin_value = num_bins
                    # replace data set value with binned value
                    new_frame.loc[row_num, column_label] = bin_value
                    row_num = row_num + 1

                bin_index = bin_index + 1

        # return discretized data frame
        return new_frame
	

    # SCRAMBLING


    # return data frame where scramble_factor % of the features are scrambled
    def get_scrambled_data_frame(self, data_frame, scramble_factor):
    	num_features = len(list(data_frame.columns.values)) - 1 # -1 to ignore CLASS column
    	# calculate number of features to scramble based on scramble factor
    	num_scrambled_features = math.ceil(num_features * scramble_factor)
    	column_indexes_scrambled = []
    	scrambled_data_frame = data_frame.copy()

    	# randomly generate list of indexes to scramble, equal to size of num_scrambled_features
    	for column_index in range(num_scrambled_features):
    		scramble_index = random.randint(0, num_features)
    		while scramble_index in column_indexes_scrambled:
    			scramble_index = random.randint(0, num_features)
    		column_indexes_scrambled.append(scramble_index)


    	for scrambled_column_index in column_indexes_scrambled:
            try:
        		# convert scrambled column index value (numeric) to column name (string)
                scrambled_column_index_str = str(self.get_column_name_from_column_index(data_frame, scrambled_column_index))
                # get list of column indexes that are randomly shuffled with respect to original list of column indexes
                scrambled_column_indexes = self.get_scrambled_column_indexes(data_frame[scrambled_column_index_str])
                # scramble rows in scrambled_column_index_str column based on scrambled_column_indexes list
                for old_index, random_index in enumerate(scrambled_column_indexes):
                    scrambled_data_frame.loc[old_index, scrambled_column_index_str] = data_frame.loc[random_index, scrambled_column_index_str]
            except:
                # TODO: figure out why these errors are happening and fix whatever the issue is
                print('ERROR: get_scrambled_data_frame: scrambled_column_index => ' + scrambled_column_index_str)
                pass

    	# return data_frame with percentage of columns scrambled
    	return (scrambled_data_frame, column_indexes_scrambled)


    # return list of scrambled column indexes, randomly scrambled using shuffle() built-in
    def get_scrambled_column_indexes(self, column_data):
    	column_data_indexes = list(range(len(column_data)))
    	random.shuffle(column_data_indexes) # shuffle in place
    	return column_data_indexes


    # utility method: return column name given column index
    def get_column_name_from_column_index(self, data_frame, column_index):
    	return data_frame.columns.values[column_index]


    # utility method: return column index given column name
    def get_column_index_from_column_name(self, data_frame, column_name):
    	return data_frame.columns.get_loc(column_name)


# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running preprocessor...')

    '''    
    data_api_impl = DataApi('../../data/')
    preprocess_impl = Preprocess()
    
    breast_cancer_data = data_api_impl.get_breast_cancer_data()
    glass_data = data_api_impl.get_glass_data()
    iris_data = data_api_impl.get_iris_data()
    house_votes_data = data_api_impl.get_house_votes_data()

    #preprocess_impl.preprocess_breast_cancer_data(breast_cancer_data)
    
    #print('The following are the preprocessed data sets: ')
    print(preprocess_impl.preprocess_iris_data(iris_data))
    print((preprocess_impl.preprocess_glass_data(glass_data)))
    #print(test.replace_missing_values_house_votes(house_votes_data))
    #print(test.replace_missing_values_breast_cancer())
    '''

