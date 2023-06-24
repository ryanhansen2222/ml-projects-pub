#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../data')
sys.path.append('../networks')
sys.path.append('../logging')
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/utilities')

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from neural_network import NeuralNetwork
from mlp_network import MLPNetwork
from rbf_network import RBFNetwork
from logger import Logger
from utils import Utils

import numpy as np
import pickle
from datetime import datetime


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
        self.logger = Logger('DEBUG') # configure log level here

        # datalayer instance - read csv data files and convert into raw data frames
        self.datalayer = DataApi('../../data/')
        # preprocessor instance - everything for prerocessing data frames
        self.preprocessor = Preprocessor()
        # cross_validator instance - setup cross validation partitions
        self.cross_validator = CrossValidator()
        # utils instance - random things
        self.utils = Utils()
        # boolean indicating whether to persist trained networks to file for reuse
        self.save_network_instance = True


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
        self.logger.log('DEMO', 'data_set_name: \t%s\n' % str(data_set_name))
        self.logger.log('DEMO', 'raw data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('DEMO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        data = self.preprocessor.preprocess_raw_data_frame(data, data_set_name)
        self.logger.log('DEMO', 'preprocessed data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('DEMO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        return data


    '''
    run experiment

    INPUT:
        - data_set_name: name of data set to run experiment on
        - neural_network: instance of neural network to train/test with data
        - hyperparams: hyperparameters and corresponding values to use in experiment

    OUTPUT:
        - <void> - logs all the important stuff at DEMO level
    '''
    def run_experiment(self, data_set_name, neural_network, hyperparams):

        # LAYER ACTIVATION FUNCTION SPECIFICATION

        self.logger.log('DEMO', 'hyperparams: \n%s' % self.get_hyperparams_str(hyperparams))

        # DATA RETRIEVAL AND PREPROCESSING

        data = self.get_experiment_data(data_set_name)

        self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name))
        
        # CROSS VALIDATION PARTITIONING

        # get cross validation partitions for data
        cv_partitions = self.cross_validator.get_cv_partitions(data)

        # dictionary for storing accuracy results
        cv_results = {}
        # list to store amount of accuracy/error improvement for each cross validation partition
        improvement_vals = []
        # list of sizes of test sets used for getting average test set size
        test_data_sizes = []

        # NEURAL NETWORK TRAINING AND TESTING

        for partition in cv_partitions:
            # initialize key and corresponding nested dictionary in results dictionary
            test_data_key = 'test_data_' + str(partition)
            cv_results[test_data_key] = {}
            # get training set and test set for given cross validation partition
            train_data, test_data = cv_partitions[partition]
            test_data_sizes.append(test_data.shape[0]) # add number of rows in test set to test_set_sizes list

            # HANDLE RBF NETWORK P2 RESULTS

            if neural_network.network_name == 'RBF':
                # configure RBF network shape based on training data
                neural_network.configure_rbf_network(train_data, data, data_set_name, hyperparams["k"])

            # GRADIENT DESCENT

            # run gradient descent for given neural network instance
            test_result_vals = neural_network.train_gradient_descent(train_data, hyperparams, partition, test_data)

            improvement = abs(test_result_vals[-1] - test_result_vals[0])
            improvement_vals.append(improvement)

            self.logger.log('DEMO', ('accuracy improvement: %s\n\n' if neural_network.CLASSIFICATION else \
                                     'error reduction: %s\n\n') % str(improvement), True)

            # append accuracy/error result of final gradient descent iteration to results dictionary
            cv_results[test_data_key] = test_result_vals[-1]

            # only do the first CV iteration if we're just going to save the MLP network for the extra credit project
            if self.save_network_instance == True and neural_network.network_name == 'MLP':
                # use pickle module to save network instance as file for reuse later on
                self.save_network(neural_network, data_set_name, hyperparams)
                break

        # FINAL RESULTS (THE MODEL)

        self.logger.log('DEMO', '------------------------------------------------------------' \
                + ' TRAINING DONE ------------------------------------------------------------')

        self.logger.log('DEMO', 'trained network: weights --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.weights), str(self.utils.get_shapes(neural_network.weights))), True)

        self.logger.log('DEMO', 'trained network: biases --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.biases), str(self.utils.get_shapes(neural_network.biases))), True)

        self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name), True)

        self.logger.log('DEMO', 'trained network: AVERAGE ' \
            + ('ACCURACY' if neural_network.CLASSIFICATION else 'ERROR') + ' --> %s\n' \
            % str(self.get_avg_result(cv_results)), True)

        avg_improvement = sum(improvement_vals) / len(improvement_vals)
        self.logger.log('DEMO', 'average improvement: %s\n' % str(avg_improvement), True)


    '''
    use pickle module to save network instance to file
    '''
    def save_network(self, neural_network, data_set_name, hyperparams):
        path_prefix = '../../../extra-credit/data/networks/' + data_set_name + '/'
        file_name = path_prefix + data_set_name + self.get_file_name(hyperparams) + '.pkl'
        self.logger.log('DEMO', 'saving network instance with file name: %s\n' % str(file_name))

        with open(file_name, 'wb') as output:
            pickle.dump(neural_network, output, pickle.HIGHEST_PROTOCOL)


    # get file name using hyperparams
    def get_file_name(self, hyperparams):
        file_name = '_'
        
        for layer_size in hyperparams['layer_sizes_display']:
            file_name += str(layer_size) + '-'

        file_name += 'iter-' + str(hyperparams['max_iterations']) + '-'
        file_name += 'basz-' + str(hyperparams['batch_size']) + '-'
        file_name += 'eta-' + str(hyperparams['eta']) + '-'
        file_name += 'mo-' + str(hyperparams['use_momentum']) + '-'
        file_name += 'mbe-' + str(hyperparams['momentum_beta']) + '-'

        file_name += 't-' + str(int(datetime.timestamp(datetime.now())))

        return file_name


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

    print('\nrunning Experiment from ExperimentRunner...\n')

    experiment_runner = ExperimentRunner()


    # DATA SET CONFIGURATION

    # CHANGE HERE: specify data set name
    data_set_name = 'abalone'

    # NETWORK INSTANTIATION

    '''
    the implementation allows for an arbitrary number of inputs/outputs,
    but the networks must have logical i/o shapes based on each data set.

    note the following requirements for network shapes:

    CLASSIFICATION:
        - segmentation: 19 inputs, 7 outputs
        - car: 6 inputs, 4 outputs
        - abalone: 8 inputs, 28 outputs

    REGRESSION:
        - machine: 9 inputs, 1 output (regression)
        - forest fires: 12 inputs, 1 output (regression)
        - wine: 11 inputs, 1 output (regression)
    '''

    # CHANGE HERE: create MLP neural network instance
    neural_network = MLPNetwork(data_set_name, [8, 28]) # abalone mlp network
    #neural_network = MLPNetwork(data_set_name, [6, 4, 2, 4]) # car mlp network
    #neural_network = MLPNetwork(data_set_name, [19, 7]) # segmentation mlp network
    #neural_network = MLPNetwork(data_set_name, [9, 3, 1]) # machine mlp network
    #neural_network = MLPNetwork(data_set_name, [12, 6, 1]) # forest fires mlp network
    #neural_network = MLPNetwork(data_set_name, [11, 1]) # wine mlp network

    # CHANGE HERE: create RBF neural network instance
    #neural_network = RBFNetwork(data_set_name, [8, 0, 28], 'cnn') # abalone rbf network
    #neural_network = RBFNetwork(data_set_name, [6, 0, 4], 'enn') # car rbf network
    #neural_network = RBFNetwork(data_set_name, [19, 0, 7], 'enn') # segmentation rbf network
    #neural_network = RBFNetwork(data_set_name, [9, 0, 1], 'enn') # machine rbf network
    #neural_network = RBFNetwork(data_set_name, [12, 0, 1], 'enn') # forest fires rbf network
    #neural_network = RBFNetwork(data_set_name, [11, 0, 1], 'enn') # wine rbf network

    # HYPERPARAMETERS

    hyperparams = {}

    # CHANGE HERE: configure training parameters for gradient descent
    hyperparams['max_iterations'] = 10
    hyperparams['batch_size'] = 10
    hyperparams['eta'] = 4

    # CHANGE HERE: configure activation functions for each layer, options: ['sigmoid', 'relu', 'tanh']
    hyperparams['layer_activation_funcs'] = ['sigmoid' for layer_idx in range(len(neural_network.layer_sizes)-1)]
    #hyperparams["layer_activation_funcs"][-1] = 'sigmoid' # use sigmoid for output layer
    hyperparams['layer_sizes_display'] = neural_network.layer_sizes

    if neural_network.network_name == 'RBF':
        # DO NOT change this line here - this is not a config line, just used for better demo logging
        hyperparams['layer_activation_funcs'] = ['rbf', hyperparams['layer_activation_funcs'][-1]]

    # CHANGE HERE: configure whether momentum should be used in training
    hyperparams['use_momentum'] = True
    hyperparams['momentum_beta'] = 0.9 # commonly used value for momentum beta

    # RBF NETWORK SPECIFIC CONFIG

    # CHANGE HERE: k value for k nearest neighbor and variants from P2
    hyperparams['k'] = 10


    # RUN EXPERIMENT

    experiment_runner.run_experiment(data_set_name, neural_network, hyperparams)


