#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')

import pandas as pd
from data_api import DataApi


# CLASS


'''
    This class takes in a dictionary of predictions for a given test set, and calculates 
    a zero/one loss function value and mean squared error for the predictions by 
    comparing them to the real class values for the given test rows.
'''


class ResultsProcessor:


    def __init__(self):
        self.DEBUG = False
        # get instance of DataApi for access to data
        self.data_api_impl = DataApi('../../data/')


    # calculate zero/one loss value and mean squared error for given set of predictions
    def loss_function_analysis(self, data_frame, data_frame_name, predictions_dict):
        count = 0
        right = 0
        wrong = 0
        for x in predictions_dict:
            count += 1
            actual_class_value = self.data_api_impl.get_class_value_for_data_frame_row(data_frame, int(x))
            if actual_class_value == predictions_dict[x]:
                right += 1
            else:
                wrong += 1

        # Mean Squared Error
        # If we get the expected value we have a 0
        # Else we get a 1
        # Sum these values
        # Square them
        # Divide by total number of tests
        mse = wrong / count

        # return tuple of loss function values (zero_one_loss, mean_squared_error)
        return (wrong, mse)


    # get average loss function values (zero/one loss and mean squared error) for given cross validation results
    def get_avg_loss_vals(self, cross_validation_results):
        zero_one_loss_vals = []
        mean_squared_error_vals = []
        # for each cross validation partition, append loss function values to corresponding lists
        for test_set_key in cross_validation_results:
            test_set_results = cross_validation_results[test_set_key]
            zero_one_loss_vals.append(test_set_results['zero_one_loss'])
            mean_squared_error_vals.append(test_set_results['mean_squared_error'])

        # should always be 10 since we're always doing 10-fold cross validation
        test_set_count = len(cross_validation_results)
        # calculate average values
        avg_zero_one_loss = sum(zero_one_loss_vals) / test_set_count
        avg_mean_squared_error = sum(mean_squared_error_vals) / test_set_count

        # return tuple with average values for zero_one_loss and mean_squared_error
        return (avg_zero_one_loss, avg_mean_squared_error)


if __name__ == "__main__":

    print('running results processor...')
