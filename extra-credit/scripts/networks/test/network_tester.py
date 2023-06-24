#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
# P2 imports
sys.path.append('../../../../project-2/scripts/data_api')
sys.path.append('../../../../project-2/scripts/preprocessing')
sys.path.append('../../../../project-2/scripts/utilities')
sys.path.append('../../../../project-2/scripts/cross_validator')
# P3 imports
sys.path.append('../../../../project-3/scripts/networks')
sys.path.append('../../../../project-3/scripts/logging')
# P4 imports
sys.path.append('../../../../project-4/scripts/learners')

# library imports
import pandas as pd
import numpy as np
import pickle

# P2 imports
from data_api import DataApi
from preprocessor import Preprocessor
from utilities import Utilities
from cross_validator import CrossValidator

# P3 imports
from mlp_network import MLPNetwork
from logger import Logger
from utils import Utils

# P4 imports
from pso_learner import ParticleSwarmOptimizationLearner


# CLASS


'''
    This class is responsible for testing pickled/unpickled networks.
'''


class NetworkTester():


    '''
    CONSTRUCTOR

    args:
    '''
    def __init__(self):
        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('INFO')


# SCRIPT-LEVEL HELPER METHODS


# get average result given cross validation results dictionary
def get_avg_result(cv_results):
    result_vals = []
    # for each cross validation partition, append result value to corresponding list
    for test_data_key in cv_results:
        test_result = cv_results[test_data_key]
        result_vals.append(test_result)

    # should always equal the value of the 'folds' variable in cross validator
    test_data_count = len(cv_results)
    # calculate average values
    avg_result = sum(result_vals) / test_data_count
    # return average result
    return avg_result



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning NetworkTester...\n')

    data_set_name = 'car'

    logger = Logger('INFO')
    cross_validator = CrossValidator()
    data_api = DataApi('../../../../project-3/data/')
    preprocessor = Preprocessor()
    utils = Utils()

    # read in data from csv file and do preprocessing
    data = data_api.get_raw_data_frame(data_set_name)
    data = preprocessor.preprocess_raw_data_frame(data, data_set_name)

    with open('../../../data/networks/car/car_6-4-2-4-iter:15-basz:10-eta:5-mo:True-mbe:0.9-t:1575616639.pkl', 'rb') as input:
        # read neural network instance from file into local variable
        neural_network = pickle.load(input)
        # verify shape of deserialized network instance
        assert neural_network.layer_sizes == [6, 4, 2, 4]

        # dictionary for storing accuracy results
        cv_results = {}
        # get cross validation partitions for data
        cv_partitions = cross_validator.get_cv_partitions(data)

        # NETWORK TESTING

        for partition in cv_partitions:

            # initialize key and corresponding nested dictionary in results dictionary
            test_data_key = 'test_data_' + str(partition)
            cv_results[test_data_key] = {}

            # get training set and test set for given cross validation partition
            _, test_data = cv_partitions[partition]

            # append accuracy/error result to dictionary for logging
            cv_results[test_data_key] = neural_network.do_test_data(partition, 0, test_data)

        # LOG TEST RESULTS AFTER ALL CROSS VALIDATION PARTITIONS

        logger.log('INFO', 'unpickled network: weights --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.weights), str(utils.get_shapes(neural_network.weights))), True)

        logger.log('INFO', 'unpickled network: biases --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.biases), str(utils.get_shapes(neural_network.biases))), True)

        logger.log('INFO', 'data_set_name: %s\n' % str(data_set_name), True)

        logger.log('INFO', 'unpickled network: AVERAGE ' \
            + ('ACCURACY' if neural_network.CLASSIFICATION else 'ERROR') + ' --> %s\n' \
            % str(get_avg_result(cv_results)), True)


