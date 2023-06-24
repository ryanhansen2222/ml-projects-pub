#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../algorithms')
sys.path.append('../cross_validator')
sys.path.append('../../../data')

from data_api import DataApi
from k_nearest_neighbor import KNN
from cross_validator import CrossValidator
from preprocessor import Preprocessor

import math
import random
import pandas as pd


# CLASS

'''
    This class handles all things edited knn. It inherits from the parent KNN class,
    in order to reuse the do_knn() method implemented in the KNN class.
'''


class EditedKNN(KNN):


    def __init__(self):
        KNN.__init__(self)
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')


    '''
    get edited training set using kNN

    INPUT:
        - raw_train_set: full training set (before editing out values)

    OUTPUT:
        - edited training set with misclassified rows removed
    '''
    def get_edited_training_set(self, original_train_set, distance_matrix, k):
        # make copy of original training set dataframe so we can drop rows from the copy
        edited_train_set = original_train_set.copy()
        '''
        call knn_predict() to get dictionary of predictions, key is test instance index
            - in this case, the 'test set' is actually the train set,
                since we're classifying each training point using all the other training points
        '''
        predictions = self.knn_predict(original_train_set, original_train_set, k, distance_matrix)

        if self.DEBUG:
            print('\nPREDICTIONS:\n')
            print(predictions)

        # for each point in the training set
        for instance_idx in predictions:
            # get our prediction for the class of the instance at index
            prediction = predictions[instance_idx][0]
            # get the actual class value of the instance at index
            actual = predictions[instance_idx][1]

            should_drop = False
            if self.CLASSIFICATION:
                if prediction != actual:
                    should_drop = True
            elif self.REGRESSION:
                diff = math.pow(prediction - actual, 2)
                #print('ENN: diff: ' + str(diff))
                if diff > 100:
                    should_drop = True

            if should_drop:
                # if the prediction is not the actual, drop the instance from the dataframe
                edited_train_set.drop(int(instance_idx), axis=0, inplace=True)
                if self.DEBUG:
                    print('! enn: dropped instance_idx --> ' + instance_idx + ' !')

        if self.DEBUG:
            print('edited_train_set after removals:')
            print(edited_train_set)

        # return edited training set dataframe, consisting of points that were classified correctly
        return edited_train_set


    '''
    run knn on training set / test set combination for given data frame, after editing training set

    INPUT:
        - train_data: training data that we will edit before running knn
        - test_data: test data
        - data_frame: full dataframe
        - k: value for k parameter

    OUTPUT:
        - return a dictionary where key is the instance index and value is a tuple: (prediction, actual)
    '''
    def do_enn(self, train_data, test_data, data_frame, k):
        # get matrix of distances from each point to every other point in dataframe
        feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']

        if self.DEBUG:
            print('get_edited_training_set: getting distance matrix for original train set feature vectors:')
            print(feature_vectors_df)

        distance_matrix = self.get_distance_matrix(feature_vectors_df)
        # get the edited training set by removing rows that were misclassified with knn
        edited_train_data = self.get_edited_training_set(train_data, distance_matrix, k)

        print('ENN: original train data shape: ' + str(train_data.shape))
        print('ENN: edited train data shape: ' + str(edited_train_data.shape))

        '''
        get dictionary of predictions, key is test instance index
        in this case, we're calling knn_predict() using the edited training set
        '''
        predictions = self.knn_predict(edited_train_data, test_data, k, distance_matrix)

        if self.DEBUG:
            print('\nPREDICTIONS:\n')
            print(predictions)

        # return dictionary where key is the instance index and value is a tuple: (prediction, actual)
        return predictions


    '''
    get edited training set for use by radial basis function network class RBFNetwork in project-3

    INPUT:
        - train_data: training data that we will edit using knn
        - data_frame: full dataframe
        - k: value for k parameter

    OUTPUT:
        - data frame that consists of the instances in the editing training set
    '''
    def get_enn_for_rbf_network(self, train_data, data_frame, k):
        # get matrix of distances from each point to every other point in dataframe
        feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']
        # get distance matrix for feature vectors data frame
        distance_matrix = self.get_distance_matrix(feature_vectors_df)
        # return the edited training set by removing rows that were misclassified with knn
        return self.get_edited_training_set(train_data, distance_matrix, k)



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running edited knn...')
    edited_knn = EditedKNN()

    data_api_impl = DataApi('../../data/')
    cross_validator_impl = CrossValidator()
    preprocessor_impl = Preprocessor()

    wine_data = data_api_impl.get_raw_data_frame('wine')
    prep_wine_data = preprocessor_impl.preprocess_raw_data_frame(wine_data, 'wine')

    wine_data_train_set = cross_validator_impl.get_training_set(prep_wine_data, test_set_number=3)
    print('wine_data_train_set.shape: ' + str(wine_data_train_set.shape))

    edited_train_set = edited_knn.get_edited_training_set(wine_data_train_set, k=25)
    print('edited_train_set.shape: ' + str(edited_train_set.shape))
