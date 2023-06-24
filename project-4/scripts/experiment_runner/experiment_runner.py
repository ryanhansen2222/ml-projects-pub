#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../learners')
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
        self.logger = Logger('DEMO') # configure log level here

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
            - contains string indicating which learning algorithm to use to train network

    OUTPUT:
        - <void> - logs all the important stuff at DEMO level
    '''
    def run_experiment(self, data_set_name, neural_network, hyperparams):

        # LAYER ACTIVATION FUNCTION SPECIFICATION

        #self.logger.log('DEMO', 'hyperparams: \n%s' % self.get_hyperparams_str(hyperparams))

        # DATA RETRIEVAL AND PREPROCESSING

        data = self.get_experiment_data(data_set_name)

       # self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name))

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

            # LEARNING

            # run configured learning algorithm to learn weights/biases for given neural network
            test_result = neural_network.learn(train_data, hyperparams, partition, test_data)

            if isinstance(test_result, list):
                # just use last result if test_result is list
                test_result = test_result[-1]

            # append accuracy/error result of best individual to results dictionary
            cv_results[test_data_key] = test_result

        # FINAL RESULTS (THE MODEL)

        self.logger.log('DEMO', '------------------------------------------------------------' \
               + ' TRAINING/TESTING DONE ------------------------------------------------------------')

        '''
        self.logger.log('DEMO', 'trained network: weights --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.weights), str(self.utils.get_shapes(neural_network.weights))), True)

        self.logger.log('DEMO', 'trained network: biases --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.biases), str(self.utils.get_shapes(neural_network.biases))), True)
        '''
        self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name), True)

        self.logger.log('DEMO', 'trained network: AVERAGE ' \
            + ('ACCURACY' if neural_network.CLASSIFICATION else 'ERROR') + ' --> %s\n' \
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

    print('\nrunning P4 Experiment from ExperimentRunner...\n')

    experiment_runner = ExperimentRunner()


    # DATA SET CONFIGURATION ------------------------------------------------------------------------------- DATA

    # CHANGE HERE: specify data set name --> ['abalone', 'car', 'segmentation', 'machine', 'forestfires', 'wine']
    data_set_name = 'segmentation'


    # NETWORK INSTANTIATION ----------------------------------------------------------------------------- NETWORK
    hidden_layers = [19, 7]

    # CHANGE HERE: create MLP neural network instance with specified hidden layers
    #neural_network = MLPNetwork(data_set_name, [8, 6, 3, 28]) # abalone mlp network
    #neural_network = MLPNetwork(data_set_name, [6, 4, 2, 4]) # car mlp network
    neural_network = MLPNetwork(data_set_name, [19, 7]) # segmentation mlp network
    #neural_network = MLPNetwork(data_set_name, [9, 3, 1]) # machine mlp network
    #neural_network = MLPNetwork(data_set_name, [12, 6, 1]) # forest fires mlp network
    #neural_network = MLPNetwork(data_set_name, [11, 7, 3, 1]) # wine mlp network

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


    # NETWORK CONFIGURATION ----------------------------------------------------------------------------- NETWORK

    hyperparams = {}
    hyperparams['dataname']=data_set_name
    hyperparams['layers']=hidden_layers

    # CHANGE HERE: configure training parameters for gradient descent
    hyperparams['max_iterations'] = 2
    hyperparams['batch_size'] = 20
    hyperparams['eta'] = 5

    # CHANGE HERE: configure activation functions for each layer, options: ['sigmoid', 'relu', 'tanh']
    hyperparams['layer_activation_funcs'] = ['sigmoid' for layer_idx in range(len(neural_network.layer_sizes)-1)]
    #hyperparams["layer_activation_funcs"][-1] = 'sigmoid' # use sigmoid for output layer
    hyperparams['layer_sizes_display'] = neural_network.layer_sizes

    # CHANGE HERE: configure whether momentum should be used in training
    hyperparams['use_momentum'] = True
    hyperparams['momentum_beta'] = 0.9 # commonly used value for momentum beta


    # LEARNING ALGORITHM CONFIGURATION ----------------------------------------------------------------- LEARNING

    learning_hyperparams = {}

    # CHANGE HERE: configure which learning algorithm to use --> ['BPG', 'GA', 'DE', 'PSO']
    learning_hyperparams['algorithm'] = 'PSO'

    # GENETIC ALGORITHM CONFIGURATION ------------------------------------------------------------------------ GA

    learning_hyperparams['population_size'] = 4
    learning_hyperparams['generations'] = 4


    # DIFFERENTIAL EVOLUTION CONFIGURATION ------------------------------------------------------------------- DE

    learning_hyperparams['population_size'] = 10
    learning_hyperparams['max_de_iterations'] = 5
    learning_hyperparams['crossover_rate'] = .5
    #learning_hyperparams['mutation_rate'] = .05
    learning_hyperparams['scaling_factor'] = 1.2


    # PARTICLE SWARM OPTIMIZATION CONFIGURATION ------------------------------------------------------------- PSO

    learning_hyperparams['n_particles'] = 4


    # RUN EXPERIMENT --------------------------------------------------------------------------------------------

    #ONLY FOR PSO or GA
    hyperparams['learning_hyperparams'] = learning_hyperparams

    experiment_runner.run_experiment(data_set_name, neural_network, hyperparams)
