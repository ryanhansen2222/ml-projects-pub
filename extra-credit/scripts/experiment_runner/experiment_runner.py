#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../networks')
# P4 imports
sys.path.append('../../../project-4/scripts/learners')
# P3 imports
sys.path.append('../../../project-3/data')
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')
# P2 imports
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/utilities')

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from neural_network import NeuralNetwork
from mlp_network import MLPNetwork
from sae_network import SAENetwork
from logger import Logger
from utils import Utils

import numpy as np


# CLASS

'''
    This class handles everything for configuring and running experiments.
'''

class ExperimentRunner:


    '''
    CONSTRUCTOR
    '''
    def __init__(self):
        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('INFO') # configure log level here

        # datalayer instance - read csv data files and convert into raw data frames
        self.datalayer = DataApi('../../../project-3/data/')
        # preprocessor instance - everything for prerocessing data frames
        self.preprocessor = Preprocessor()
        # cross_validator instance - setup cross validation partitions
        self.cross_validator = CrossValidator()
        # utils instance - random things
        self.utils = Utils()


    # get average result given cross validation results dictionary
    def get_avg_result(self, cv_results):
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


    '''
    get preprocessed data ready for consumption by experiment running logic

    INPUT:
        - data_set_name: name of data set to fetch data for

    OUTPUT:
        - preprocessed data frame - fully ready for experiment consumption
    '''
    def get_experiment_data(self, data_set_name):
        data = self.datalayer.get_raw_data_frame(data_set_name)
        self.logger.log('INFO', 'data_set_name: \t%s\n' % str(data_set_name))
        self.logger.log('INFO', 'raw data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('INFO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        data = self.preprocessor.preprocess_raw_data_frame(data, data_set_name)
        self.logger.log('INFO', 'preprocessed data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('INFO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        return data


    '''
    run experiment

    INPUT:
        - data_set_name: name of data set to run experiment on
        - sae_neural_network: instance of SAE neural network to train/test with data
        - hyperparams: hyperparameters and corresponding values to use in experiment
            - contains string indicating which learning algorithm to use to train network

    OUTPUT:
        - <void> - logs all the important stuff at DEMO level
    '''
    def run_experiment(self, data_set_name, sae_neural_network, hyperparams):

        # LAYER ACTIVATION FUNCTION SPECIFICATION

        self.logger.log('INFO', 'hyperparams: \n%s' % self.get_hyperparams_str(hyperparams))

        # DATA RETRIEVAL AND PREPROCESSING

        data = self.get_experiment_data(data_set_name)

        self.logger.log('INFO', 'data_set_name: %s\n' % str(data_set_name))
        
        # CROSS VALIDATION PARTITIONING

        # get cross validation partitions for data
        cv_partitions = self.cross_validator.get_cv_partitions(data)

        # dictionary for storing accuracy results
        cv_results = {}
        # list to store amount of average MSE improvement for each cross validation partition
        improvement_vals = []
        # list of sizes of test sets used for getting average test set size
        test_data_sizes = []

        # SAE NEURAL NETWORK TRAINING AND TESTING

        for partition in cv_partitions:
            self.logger.log('INFO', 'starting cv partition: %s...\n' % str(partition+1), True)
            # initialize key and corresponding nested dictionary in results dictionary
            test_data_key = 'test_data_' + str(partition)
            cv_results[test_data_key] = {}
            # get training set and test set for given cross validation partition
            train_data, test_data = cv_partitions[partition]
            test_data_sizes.append(test_data.shape[0]) # add number of rows in test set to test_set_sizes list

            # LEARNING

            # train SAE network using unsupervised pre-training for AE layers, and backprop for full network
            sae_neural_network.learn(train_data, hyperparams, partition, test_data)

            # TESTING

            # test full SAE network (with MLP network on top for prediction) and calculate accuracy/error result
            test_result = sae_neural_network.predict(train_data, hyperparams, partition, test_data)

            # append accuracy/error result to dictionary of results
            cv_results[test_data_key] = test_result

            # RESET

            sae_neural_network.reset_ae_layers()

            # only do one CV partition if wanting to get idea of results for development/debugging
            if self.exit_early:
                break

        # FINAL RESULTS (THE MODEL)

        self.logger.log('INFO', '------------------------------------------------------------' \
                + ' TRAINING/TESTING DONE ------------------------------------------------------------')

        self.logger.log('INFO', 'trained network: weights --> \n\n%s, shapes: %s\n' \
            % (str(sae_neural_network.weights), str(self.utils.get_shapes(sae_neural_network.weights))), True)

        self.logger.log('INFO', 'trained network: biases --> \n\n%s, shapes: %s\n' \
            % (str(sae_neural_network.biases), str(self.utils.get_shapes(sae_neural_network.biases))), True)

        self.logger.log('INFO', 'data_set_name: %s\n' % str(data_set_name), True)

        self.logger.log('INFO', 'trained network: AVERAGE ' \
                        + ('ACCURACY' if sae_neural_network.CLASSIFICATION else 'ERROR') + ' --> %s\n' \
                        % str(self.get_avg_result(cv_results)), True)


    '''
    return print-friendly string of hyperparams and corresponding values

    INPUT:
        - dictionary of hyperparams as configured in script execution block below

    OUTPUT:
        - print-friendly string to be printed at start of every experiment
    '''
    def get_hyperparams_str(self, hyperparams):
        hyperparams_str = ''
        for param in hyperparams:
            hyperparams_str += '\t' + param + ': ' + str(hyperparams[param]) + '\n'
        return hyperparams_str



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning EC Experiment from ExperimentRunner...\n')

    experiment_runner = ExperimentRunner()


    # DATA SET CONFIGURATION ------------------------------------------------------------------------------- DATA

    # CHANGE HERE: specify data set name --> ['abalone', 'car', 'segmentation', 'machine', 'forestfires', 'wine']
    data_set_name = 'machine'


    # NETWORK INSTANTIATION ----------------------------------------------------------------------------- NETWORK

    # CHANGE HERE: create SAE neural network instance with specified hidden layers
    #sae_neural_network = SAENetwork(data_set_name, [6, 3]) # abalone sae network
    #sae_neural_network = SAENetwork(data_set_name, [4]) # car sae network
    #sae_neural_network = SAENetwork(data_set_name, [14]) # segmentation sae network
    sae_neural_network = SAENetwork(data_set_name, [6]) # machine sae network
    #sae_neural_network = SAENetwork(data_set_name, [9]) # forest fires sae network
    #sae_neural_network = SAENetwork(data_set_name, [8]) # wine sae network


    # NETWORK CONFIGURATION ----------------------------------------------------------------------------- NETWORK

    hyperparams = {}

    # CHANGE HERE: configure training parameters for gradient descent
    hyperparams['max_iterations'] = 10
    hyperparams['batch_size'] = 4
    hyperparams['eta'] = 2

    # CHANGE HERE: configure activation functions for each layer, options: ['sigmoid', 'relu', 'tanh']
    hyperparams['layer_activation_funcs'] = ['sigmoid' for layer_idx in range(len(sae_neural_network.layer_sizes)-1)]
    #hyperparams["layer_activation_funcs"][-1] = 'sigmoid' # use sigmoid for output layer
    hyperparams['layer_sizes_display'] = sae_neural_network.layer_sizes

    # CHANGE HERE: configure whether momentum should be used in training
    hyperparams['use_momentum'] = False
    hyperparams['momentum_beta'] = 0.9 # commonly used value for momentum beta


    # RUN EXPERIMENT --------------------------------------------------------------------------------------------

    experiment_runner.exit_early = True # boolean indicating whether to only do one CV partition for dev/debug

    experiment_runner.run_experiment(data_set_name, sae_neural_network, hyperparams)


