#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../utilities')
sys.path.append('../cross_validator')
sys.path.append('../preprocessing')
sys.path.append('../../../data')

from data_api import DataApi
from distance_functions import DistanceFunctions
from cross_validator import CrossValidator
from preprocessor import Preprocessor

import time

import pandas as pd
import numpy as np
import statistics as stats

from random import shuffle
from operator import itemgetter
from statistics import StatisticsError
from scipy.spatial.distance import pdist, squareform


# CLASS

'''
    This class handles all things k nearest neighbor.
'''


class KNN:


    def __init__(self):
        self.DEBUG = False
        self.VERBOSE = False

        self.data_api_impl = DataApi('../../../data/')
        self.data_set = None

        self.CLASSIFICATION = True
        self.REGRESSION = False

        self.algorithm_name = None


    '''
    run knn on training set / test set combination for given data frame and k param value

    INPUT:
        - train_data: training data
        - test_data: test data
        - data_frame: full data frame
        - k: value for k parameter

    OUTPUT:
        - returns a dictionary where key is the instance index and value is a tuple: (prediction, actual)
    '''
    def do_knn(self, train_data, test_data, data_frame, k):

        if self.DEBUG and self.VERBOSE:
            print('\nrunning do_knn() ...\n')
            print('\n\n\ndo_knn: train_data.shape:')
            print(train_data.shape)
            print('do_knn: test_data.shape:')
            print(test_data.shape)
            print('do_knn: data_frame.shape:')
            print(data_frame.shape)
            print('train_data:')
            print(train_data)
            print('test_data:')
            print(test_data)
            print('do_knn: train_data.index:')
            print(list(train_data.index.values))
            print('do_knn: test_data.index:')
            print(list(test_data.index.values))
            print('do_knn: k param value:')
            print(str(k))
            print('data_frame:')
            print(data_frame)


        distance_matrix = None

        # we need to use a different distance matrix mechanism for kmeans_knn and kmedoids_knn
        if self.algorithm_name == 'kmeans_knn':
            print('KNN: running knn using kmeans centroids as reduced data set')
            train_test_data = train_data.append(test_data, ignore_index=False)
            tt_feature_vectors_df = train_test_data.loc[:, train_test_data.columns != 'CLASS']
            distance_matrix = self.get_distance_matrix(tt_feature_vectors_df)
            self.idx_mapping = self.get_idx_mapping(distance_matrix, train_data, test_data)
        else:
            #print('KNN: running knn using algorithm_name: ' + str(self.algorithm_name))
            # get distance matrix - matrix of distances from each point to every other point
            # do not include the CLASS column in the calculations since it is not an attribute
            feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']
            distance_matrix = self.get_distance_matrix(feature_vectors_df)
            #distance_matrix = self.get_distance_matrix(train_data.loc[:, train_data.columns != 'CLASS'])

        assert distance_matrix is not None

        print('do_knn: distance_matrix.shape: ' + str(distance_matrix.shape))

        # dictionary of predictions, key is test instance index
        predictions = self.knn_predict(train_data, test_data, k, distance_matrix)

        if self.DEBUG:
            print('\n\nPREDICTIONS:\n')
            print(predictions)

        # return dictionary where key is the instance index (as string) and value is a tuple: (prediction, actual)
        return predictions


    '''
    regular k nearest neighbor

    INPUT:
        - data_frame: data set to run knn against
        - instance_index: the index of the row we want to classify
        - k: the value for the parameter k used in kNN

    OUTPUT:
        - dictionary of predictions - class or value for instances in test_data
    '''
    def knn_predict(self, train_data, test_data, k, distance_matrix):

        predictions = {}

        filter_test_indexes = list(test_data.index.values)

        # predict class or value for each instance in test data set
        for instance_idx in range(test_data.shape[0]):

          # get row data for test instance at instance index
          test_instance = test_data.iloc[instance_idx, :]

          if self.DEBUG:
              print('\n\n\nknn_predict: using instance_idx: ' + str(instance_idx))
              print('knn_predict: predicting test_instance.name: ' + str(test_instance.name))
              print('knn_predict: test_data.shape: ' + str(test_data.shape))
              print('knn_predict: train_data.shape: ' + str(train_data.shape))

          # get k nearest neighbors from instance_idx using distance matrix
          nearest_neighbors = self.get_knn_by_dist_mat(distance_matrix, test_instance.name, k, filter_test_indexes)

          # OUTPUT FOR DEMO
          if self.DEBUG:
            print('knn_predict: nearest_neighbors:')
            print(nearest_neighbors)

          # make assertion to verify we always have k nearest neighbors
          assert nearest_neighbors is not None and len(nearest_neighbors) == k

          # get class values of k nearest neighbors
          nn_vals = self.get_nn_labels_by_idx(train_data, nearest_neighbors)

          # OUTPUT FOR DEMO
          if self.DEBUG:
            print('knn_predict: k: ' + str(k) + ', nn_vals:')
            print(nn_vals)

          # get highest frequency class (mode) from list of nearest neighbor labels
          prediction = self.get_prediction_from_nn_vals(nn_vals)
          # get actual class value for instance_idx
          actual = test_instance['CLASS']
          # if doing regression, convert actual value to float for comparison to prediction
          if self.REGRESSION:
              actual = float(actual)

          if self.DEBUG:
              print('prediction --> [df_idx: ' + str(instance_idx) + \
                      ', instance.name: ' + str(test_instance.name) + \
                      ', prediction: ' + str(prediction) + \
                      ', actual: ' + str(actual) + ']' + \
                      (' --> WRONG!' if self.CLASSIFICATION and prediction != actual else ''))
                      # print MSE here when ready

          # TODO: assert prediction and actual have same types here (string, string) or (float, float)
          predictions[str(test_instance.name)] = (prediction, actual)

        # return dictionary of predictions - tuples (prediction, actual)
        return predictions


    # get prediction using either mode for classification, or mean for regression
    def get_prediction_from_nn_vals(self, nn_vals):
        if self.data_set is None:
            raise Exception('ERROR: must specify data_set in KNN')
        elif self.data_set in ['abalone', 'car', 'segmentation']:
            # CLASSIFICATION data sets - return mode of nearest neighbors
            try:
                nn_mode = self.get_nn_mode(nn_vals)
            except:
                nn_mode = 'CLASS' # workaround for now...
            return nn_mode
        elif self.data_set in ['machine', 'forestfires', 'wine']:
            # REGRESSION data sets - return mean of nearest neighbors
            return self.get_nn_mean(nn_vals)
        else:
            raise Exception('ERROR: unknown data_set: ' + str(self.data_set))


    # return the highest frequency class value from a list of nearest neighbor class values
    def get_nn_mode(self, nn_labels):
        nn_mode = None
        try:
            # call statistics library method to get mode
            nn_mode = stats.mode(nn_labels)
        except StatisticsError:
            # no unique mode, must return one of the modes by randomly selecting
            # get list of tuples where each tuple is (nn_labels_value, frequency)
            item_counts = [(i, nn_labels.count(i)) for i in set(nn_labels)]
            # sort the list of tuples by frequency in descending order
            sorted_item_counts = sorted(item_counts, key=itemgetter(1), reverse=True)
            # randomly select one of the highest frequency items (modes) and return
            return self.get_random_mode(sorted_item_counts)

        # the mode() library method worked so return that value
        return nn_mode


    # return randomly selected mode given a sorted list of item counts
    def get_random_mode(self, sorted_item_counts):
        # get frequency of highest frequency value
        highest_freq_val = sorted_item_counts[0][1]
        mode_tuples = []

        for mode_tuple in sorted_item_counts:
            # if the items frequency is not the highest we can stop iterating
            if mode_tuple[1] != highest_freq_val:
                break
            # if tuple item is a mode then add it to the list of modes
            if mode_tuple[1] == highest_freq_val:
                mode_tuples.append(mode_tuple)

        # randomly shuffle the modes so we can return a random one
        shuffle(mode_tuples)
        # return mode value from randomly shuffled list
        return mode_tuples[0][0]


    # get average value of numbers in nn_vals list
    def get_nn_mean(self, nn_vals):
        if self.DEBUG:
            print('get_nn_mean: nn_vals:')
            # print(nn_vals)

        if nn_vals:
            nn_num_vals = self.get_nn_num_vals(nn_vals)
            # print('get_nn_mean: nn_num_vals:')
            # print(nn_num_vals)

            if self.data_set in ['machine', 'wine']:
                # CLASS values are ints for machine and wine data, round to nearest int
                return round(stats.mean(nn_num_vals))
            elif self.data_set in ['forestfires']:
                # CLASS values are floats with one decimal point for forestfires data
                return stats.mean(nn_num_vals)
            else:
                raise Exception('ERROR: unexpected data set name: ' + str(self.data_set))

        else:
            print('WARN: nn_vals is empty!')
            return 0 # workaround for now


    # convert string type numbers to actual floats for mean calculation
    def get_nn_num_vals(self, str_nn_vals):
        try:
            return [float(str_nn_vals[i]) for i in range(len(str_nn_vals))]
        except:
            raise Exception('ERROR: cannot convert string nn_vals')


    '''
    get class values for list of nearest neighbors

    INPUT:
        - train_data: training data frame we're working with
        - nearest_neighbors: list of indexes of nearest neighbors

    OUTPUT:
        - list of class values for nearest neighbors in input list
    '''
    def get_nn_labels_by_idx(self, train_data, nearest_neighbors):
        if self.DEBUG and self.VERBOSE:
          print('get_nn_labels_by_idx: nearest_neighbors ARG:')
          print(nearest_neighbors)

        train_data_indexes = list(train_data.index.values)
        nn_labels = []

        if nearest_neighbors:
            for nn_idx in nearest_neighbors:
                if nn_idx in train_data_indexes:
                    # append class value to list of nearest neighbor class values
                    nn_labels.append(train_data.loc[nn_idx, 'CLASS'])
                else:
                    if self.DEBUG and self.VERBOSE:
                        print('ERROR: nn_idx not in train_data.index: ' + str(nn_idx))
                        #print('KNN: train_data_indexes:')
                        #print(train_data_indexes)
        else:
            print('ERROR: no nearest neighbors:')
            print(nearest_neighbors)
        return nn_labels


    '''
    get distance matrix using pdist and squareform methods from scipy.spatial.distance

    INPUT:
        - data_frame: data frame we're working with

    OUTPUT:
        - distance matrix containing distances between every pair of points in data frame
    '''
    def get_distance_matrix(self, data_frame):
        # for some reason this has to be done here even though it's done above...
        feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']
        # get distance matrix (upper triangle) using distance metric
        distances = pdist(feature_vectors_df.values, metric='euclidean')
        # fill in lower triangle maintaining symmetry
        dist_matrix = squareform(distances)
        # return full distance matrix
        return dist_matrix


    '''
    get k nearest neighbors using distance matrix

    INPUT:
        - data_frame: data frame we're working with
        - instance_idx: index of instance we want to run knn against
        - k: parameter value for k in knn

    OUTPUT:
        - return list of indexes of k nearest neighbors
    '''
    def get_knn_by_dist_mat(self, distance_matrix, instance_idx, k, filter_test_indexes):
        if self.DEBUG and self.VERBOSE:
            print('\n\nget_knn_by_dist_mat: distance_matrix:')
            print(distance_matrix)
            print('get_knn_by_dist_mat: distance_matrix.shape: ' + str(distance_matrix.shape))
            print('get_knn_by_dist_mat: filter_test_indexes: ')
            print(filter_test_indexes)
            print('get_knn_by_dist_mat: len(filter_test_indexes): ' + str(len(filter_test_indexes)))

        # get list of distances from instance to all other points in training set
        idx_distances = self.get_idx_distances(distance_matrix, instance_idx, filter_test_indexes)

        # return indexes of k points with least distance from instance
        knn = list(np.argpartition(idx_distances, k)[:k])

        '''
        knn = [] #initalize to length of distance matrix

        for x in range(len(distance_matrix)):
          knn.append(0)
          for y in range(k):
            if distance_matrix[instance_idx][x] < distance_matrix[instance_idx][knn[y]]:
              knn_head = knn[0:y-1]
              knn_tail = knn[y:]
              knn = knn.extend(knn_head)
              knn = knn.extend(x)
              knn = knn.extend(knn_tail)
        print("This is the knn: ")
        print(knn)
        '''
        return knn


    def get_idx_mapping(self, distance_matrix, train_data, test_data):
        idx_mapping = {}
        test_instance_idxs = []

        for instance_idx in range(test_data.shape[0]):
          # get row data for test instance at instance index
          test_instance = test_data.iloc[instance_idx, :]
          test_instance_idxs.append(test_instance.name)

        for i in range(len(test_instance_idxs)):
            test_idx_val = test_instance_idxs[i]
            idx_mapping[str(test_idx_val)] = len(train_data) + i

        return idx_mapping

        
    def get_idx_distances(self, distance_matrix, instance_idx, filter_test_indexes):

        # special handling for doing knn using k-means clustering centroids
        if self.algorithm_name == 'kmeans_knn':
            assert self.idx_mapping is not None
            instance_idx = self.idx_mapping[str(instance_idx)]
            filter_test_indexes = [self.idx_mapping[str(i)] for i in filter_test_indexes]

        # convert distances list to np array for super fast value filtering
        np_dist_list = np.array(list(distance_matrix[instance_idx]))

        if self.algorithm_name != 'enn':
            # can't just filter out the test values, the test points need to be changed to big numbers
            # that way they won't become nearest neighbors, but the indexes will still remain intact
            np_dist_list[filter_test_indexes] = 100000
        else:
            # skip test index filtering for ENN because every training point is a 'test point' in this context
            if self.DEBUG and self.VERBOSE:
                print('KNN: WARN: skipped test index filtering for ENN!')

        return list(np_dist_list)


    # HELPER METHODS


    # set data set name for context
    def set_data_set(self, data_set):
        self.data_set = data_set

        if self.data_set in ['abalone', 'car', 'segmentation']:
            # CLASSIFICATION data sets
            self.CLASSIFICATION = True
            self.REGRESSION = False
        elif self.data_set in ['machine', 'forestfires', 'wine']:
            # REGRESSION data set
            self.REGRESSION = True
            self.CLASSIFICATION = False


    # get data set name
    def get_data_set(self):
        return self.data_set


    # set algorithm name for context
    def set_algorithm_name(self, algorithm_name):
        self.algorithm_name = algorithm_name



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nk nearest neighbor...\n')

    data_api_impl = DataApi('../../data/')
    cross_validator_impl = CrossValidator()
    preprocessor_impl = Preprocessor()

    knn_impl = KNN()

    #abalone_data = data_api_impl.get_raw_data_frame('abalone')
    #abalone_data = preprocessor_impl.prep_abalone_data(abalone_data)

    '''
    # EXPERIMENT BELOW

    start_time = time.time()

    print('ABALONE DATA:')
    print('\npossible classes: ' + str(list(set(abalone_data.loc[:, 'CLASS'].values))) + '\n')

    total = 100
    k = 250

    print('k: ' + str(k) + '\n\n')

    num_correct = 0

    for i in range(total):
        prediction, actual = knn_impl.knn_predict(abalone_data, instance_idx=i, k=k)
        if prediction == actual:
            num_correct = num_correct + 1

    print('\nk: ' + str(k) + '\n')
    print('\naccuracy --> fraction: ' + str(num_correct) + '/' + str(total) + ', ratio: ' + str(num_correct / total))

    print('\n\ntotal runtime: ' + str((time.time() - start_time)/60) + ' minutes')
    '''
