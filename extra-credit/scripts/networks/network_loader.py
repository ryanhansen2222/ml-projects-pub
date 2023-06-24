#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
# P2 imports
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/utilities')
sys.path.append('../../../project-2/scripts/cross_validator')
# P3 imports
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')
# P4 imports
sys.path.append('../../../project-4/scripts/learners')

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
    This class is responsible for loading pickled/unpickled networks.
'''


class NetworkLoader():


    '''
    CONSTRUCTOR

    args:
    '''
    def __init__(self):
        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('ERROR')


    '''
    return unpickled network from file based on data set name

    INPUT:
        - data_set: name of data set to get unpickled network for

    OUTPUT:
        - return unpickled network, ready to be hooked up to SAE network for prediction
    '''
    def load_network(self, data_set):
        file_path = self.get_file_path(data_set)

        try:
            with open(file_path, 'rb') as input:
                # deserialize and return neural network instance from file
               return pickle.load(input)
        except:
            self.logger.log('ERROR', 'error unpickling neural network from file: %s' % str(file_path))
            return None


    '''
    get file path for file containing serialized network instance
    '''
    def get_file_path(self, data_set_name):
        file_path = '../../data/networks/' + data_set_name + '/'
        if data_set_name == 'segmentation':
            file_path += 'segmentation_19-7-iter-15-basz-10-eta-5-mo-True-mbe-0.9-t-1575694660.pkl'
        elif data_set_name == 'car':
            file_path += 'car_6-4-iter-15-basz-10-eta-1-mo-True-mbe-0.9-t-1575701659.pkl'
        elif data_set_name == 'abalone':
            raise Exception('no pickled network for abalone data')
        elif data_set_name == 'machine':
            file_path += 'machine_9-3-1-iter-3-basz-10-eta-5-mo-True-mbe-0.9-t-1575615328.081167.pkl'
        elif data_set_name == 'forestfires':
            file_path += 'forestfires_12-6-1-iter-15-basz-10-eta-5-mo-True-mbe-0.9-t-1575616357.pkl'
        elif data_set_name == 'wine':
            file_path += 'wine_11-1-iter-10-basz-10-eta-4-mo-True-mbe-0.9-t-1576035573.pkl'
        else:
            raise Exception('unknown data_set_name: %s' % str(data_set_name))

        return file_path



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning NetworkLoader...\n')



