#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
# P2 imports
sys.path.append('../../../project-2/data')
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/algorithms')
sys.path.append('../../../project-2/scripts/tuning')
sys.path.append('../../../project-2/scripts/results')
sys.path.append('../../../project-2/scripts/utilities')
# P3 imports
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')

import pandas as pd
import numpy as np

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from neural_network import NeuralNetwork
from logger import Logger
from utils import Utils
from network_loader import NetworkLoader


# CLASS


'''
    This class is responsible for all things stacked autoencoder.

    TODO:
        - need to figure out mechanism for "freezing" pre-trained layers
        - need to figure out mechanism for appending unpickled pre-trained MLP network
'''


class SAENetwork(NeuralNetwork):


    '''
    CONSTRUCTOR

    args:
        - data_set_name: name of data set network will operate with
        - ae_layer_sizes: list of layer sizes for auto encoder layers
            - each value denotes size of particular auto-encoder layer
    '''
    def __init__(self, data_set_name, ae_layer_sizes):
        # instantiate base class with base SAE layer sizes
        sae_base_layer_sizes = infer_layer_sizes(data_set_name)
        NeuralNetwork.__init__(self, data_set_name, sae_base_layer_sizes)

        self.network_name = 'SAE'

        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('DEMO')
        self.network_loader = NetworkLoader()

        self.ae_layer_sizes = ae_layer_sizes

        self.full_network_shape = []
        self.full_network_shape.extend(sae_base_layer_sizes)

        self.mlp_network_appended = False

        # randomly initialize weights/biases lists for base network
        self.init_weights_biases()

        # list of freeze statuses for each layer in the network - init with same size as base layers
        self.freeze_statuses = [True] * len(self.weights)
        self.freeze_statuses[-1] = False # output layer never frozen
        self.logger.log('INFO', 'len of freeze_statuses in constructor: %s' % str(len(self.freeze_statuses)))

        '''
        override weights/biases list here so we can use a more detailed structure for holding everything
        should we use a list of tuples where each tuple is (freeze-status, list) ?
            - all we need to add is the freeze-status stuff, actual content of lists will be the same
                - this might need to a ton of unnecessary overriding though.
            - just add a new list for keeping track of which indexes are frozen?
                - less overriding this way?

        should we just ditch mini-batching for this network?
        '''


    '''
    OVERRIDE

    build SAE auto-encoder layers using unsupervised pre-training

    INPUT:
        - train_data: training data to use in pre-training
        - hyperparams: dictionary of hyperparameters
        - partition: partition number
        - (optional) test_data: test data

    OUTPUT:
        - <void> SAE network is built
    '''
    def learn(self, train_data, hyperparams, partition, test_data=None):
        self.logger.log('INFO', 'base SAE network shape: %s' % str(self.layer_sizes))
        self.logger.log('INFO', 'base SAE network weights shape: %s' % str(self.utils.get_shapes(self.weights)))
        self.logger.log('INFO', 'base SAE network biases shape: %s' % str(self.utils.get_shapes(self.biases)))

        self.logger.log('DEMO', 'building SAE with unsupervised pre-training...\n')

        for ae_layer in self.ae_layer_sizes:
            self.add_layer(ae_layer, train_data, hyperparams, partition, test_data)

        self.logger.log('DEMO', 'DONE BUILDING SAE NETWORK\n', True)
        self.logger.log('DEMO', 'pre-trained SAE network shape: %s' % str(self.full_network_shape))
        #self.logger.log('DEMO', 'pre-trained SAE network shape: %s' % str(self.full_network_shape), True)
        self.logger.log('DEMO', 'pre-trained SAE network weights shape: %s' % str(self.utils.get_shapes(self.weights)))
        self.logger.log('DEMO', 'pre-trained SAE network biases shape: %s\n' % str(self.utils.get_shapes(self.biases)))


    '''
    add layer to SAE - build and train autoencoder, then append code layer to full SAE

    INPUT:
        - layer_size: size of new auto-encoder layer to add to SAE
        - train_data: training data to train new layer
        - hyperparams: dictionary of hyperparameters and corresponding values
        - cv_partition: cross validation partition number
        - (optional) test_data: test data

    OUTPUT:
        - <void> - just add the new auto-encoder layer to the SAE
    '''
    def add_layer(self, ae_layer_size, train_data, hyperparams, cv_partition, test_data):
        '''
        for each layer we need to take the current network, append the new AE layer of layer_size to the end,
        then append the output layer back on (same size as input for auto-encoding), then we do backprop with
        all the training data to fit the weights until they're good enough, and that's it.

        the only weights/biases that should be updated are the ones for the new auto-encoder layer,
        all the preceding weights/biases should be "frozen" so that they can't be updated at this time,
        unsupervised pre-training only adds/trains one layer at a time unlike regular backprop.
        '''

        '''
        for SAE network need to be able to add to the weights/biases list one layer at a time,
        and then freeze preceding layers so only the last layer gets updated in each pre-training pass.
        '''
        self.logger.log('DEMO', 'adding auto-encoder layer of size: %s ...\n' % str(ae_layer_size), True)

        output_layer = self.full_network_shape.pop(-1)
        self.full_network_shape.append(ae_layer_size)
        self.full_network_shape.append(output_layer)

        # insert false freeze status for new layer - unfrozen since we need to train it now
        self.logger.log('INFO', 'freeze_statuses before insert: %s' % str(self.freeze_statuses))
        self.freeze_statuses.insert(-1, False)
        self.logger.log('INFO', 'freeze_statuses after insert: %s' % str(self.freeze_statuses))

        # randomly initialize weights/biases for new auto-encoder layer
        self.init_layer(self.full_network_shape[-3], self.full_network_shape[-2])

        # TRAINING - train new layer using backpropagation with training data
        test_result_vals = self.train_gradient_descent(train_data, hyperparams, cv_partition, test_data)
        improvement = abs(test_result_vals[-1] - test_result_vals[0])
        self.logger.log('INFO', 'MSE improvement: %s' % str(improvement))
        self.logger.log('INFO', 'done adding layer.')

        # TESTING - test new layer to see how well it can match the input/output
        # use test_data and see what the average error diff is to confirm this is working...


    '''
    OVERRIDE

    train network with gradient descent using all batches of training data

    INPUT:
        - see base class method for detailed documentation

    OUTPUT:
        - return list of result values (accuracy/error) achieved after each iteration of gradient descent
    '''
    def train_gradient_descent(self, train_data, hyperparams, cv_partition, test_data=None):

        if not self.mlp_network_appended:
            # activation function specified by layer, options: ['sigmoid', 'relu', 'tanh']
            self.layer_activation_funcs = hyperparams["layer_activation_funcs"]
            self.layer_activation_funcs.append(self.layer_activation_funcs[-1]) # add one more for output layer

        # get values for hyperparams used in this method
        max_iterations = hyperparams["max_iterations"]
        batch_size = hyperparams["batch_size"]

        result_vals = []

        # for each iteration in gradient descent
        for iteration in range(max_iterations):
            # randomly shuffle data so the batches are different every iteration
            shuffled_train_data = train_data.sample(frac=1)
            # construct list of mini batches for current iteration
            batches = [shuffled_train_data[batch_idx:batch_idx+batch_size] \
                        for batch_idx in range(0, shuffled_train_data.shape[0], batch_size)]

            for batch in batches:
                # run backprop using given batch and update weights/biases after batch is complete
                self.do_batch(batch, hyperparams)

            if test_data is not None:
                # if test data is specified, run forward propagation using test data
                result_vals.append(self.do_test_data(cv_partition, iteration, test_data))

        self.logger.log('INFO', 'SAE train_gradient_descent: result_vals: %s' % str(result_vals))

        # return list of result values (accuracy/error) achieved for each iteration in gradient descent
        return result_vals


    '''
    randomly initialize weights/biases in new layer

    INPUT:
        - l_layer_size: left layer size
        - r_layer_size: right layer size

    OUTPUT:
    '''
    def init_layer(self, l_layer_size, r_layer_size):
        self.logger.log('INFO', 'init_layer: current network shape: %s' % str(self.full_network_shape))
        self.logger.log('INFO', 'init_layer: l_layer_size: %s, r_layer_size: %s' % (str(l_layer_size), str(r_layer_size)))

        # remove last weight matrix since it is no longer valid
        self.weights.pop(-1)
        # randomly initialize weights matrix for new layer, insert into weights list
        self.weights.append(np.random.randn(r_layer_size, l_layer_size))
        self.weights.append(np.random.randn(self.full_network_shape[-1], r_layer_size))

        # randomly initialize bias vector for new layer, insert into biases list
        self.biases.insert(-1, np.random.randn(r_layer_size, 1)) # TODO: just append?

        self.logger.log('INFO', 'weights after init_layer: %s, shape: %s' \
                        % ('SKIP', str(self.utils.get_shapes(self.weights))))
        self.logger.log('INFO', 'biases after init_layer: %s, shape: %s\n' \
                        % ('SKIP', str(self.utils.get_shapes(self.biases))))

        '''
        # zero initialize nabula gradient for new weight matrix - used in momentum calculation
        # TODO: weights list is only a single value, so do not use insert() here...
        self.prev_nab_weights.insert(-1, np.zeros(self.weights[-1].shape))
        # zero initialize nabula gradient for new biases vector - used in momentum calculation
        self.prev_nab_biases.insert(-1, np.zeros(self.biases[-1].shape))

        self.logger.log('DEBUG', 'prev_nab_weights after init_layer: %s, shape: %s' \
            % (str(self.prev_nab_weights), str(self.utils.get_shapes(self.prev_nab_weights))))
        self.logger.log('DEBUG', 'prev_nab_biases after init_layer: %s, shape: %s' \
            % (str(self.prev_nab_biases), str(self.utils.get_shapes(self.prev_nab_biases))))
        '''


    '''
    OVERRIDE

    use test_data points and log i/o mapping performance results

    INPUT:
        - cv_partition: cross validation partition number
        - iteration: iteration count in gradient descent for given cv partition
        - test_data: test data frame containing all points to predict (for cv partition)

    OUTPUT:
        - return average accuracy/error result - log at DEMO level
    '''
    def do_test_data(self, cv_partition, iteration, test_data):
        self.logger.log('DEBUG', 'do_test_data() ...')
        # list of values 0/1 where value is 1 if prediction was correct
        test_results = []
        # for each data point in the test_data dataframe
        for idx, _ in test_data.iterrows():
            # get vector of attributes for all columns except the CLASS column, cast as np.float type
            feature_vector = np.array(test_data.loc[idx, test_data.columns != 'CLASS'], dtype=np.float)
            row_vec = test_data.loc[idx, :] # full row vector

            self.logger.log('DEBUG', 'test feature vector: %s, shape: %s\n' \
                % (str(feature_vector), str(feature_vector.shape)))

            # reshape feature vector into column vector with length equal to number of attributes
            feature_vector = feature_vector.reshape(test_data.shape[1]-1, 1)

            self.logger.log('DEBUG', 'test feature column: %s, shape: %s\n' \
                % (str(feature_vector), str(feature_vector.shape)))

            # do forward propagation to calculate activations
            _, activations = self.forward_prop(feature_vector)
            # append test result to list of test results, compare output to input
            test_results.append(self.handle_prediction(idx, activations[-1], feature_vector, row_vec))

        # get average mean squared error between output and input feature vector
        avg_test_result = round(sum(test_results) / len(test_results), 5) # round to 5 decimals

        if iteration != None:
            if self.mlp_network_appended:
                # logging accuracy/error results for predictions made with MLP network appended
                if self.CLASSIFICATION:
                    self.logger.log('DEMO', 'cv_partition: %s, iteration: %s, accuracy: %s' \
                        % (str(cv_partition+1), str(iteration+1), str(avg_test_result)))
                elif self.REGRESSION:
                    self.logger.log('DEMO', 'cv_partition: %s, iteration: %s, error: %s' \
                        % (str(cv_partition+1), str(iteration+1), str(avg_test_result)))
            else:
                # logging mean squared error for i/o autoencoder mapping
                self.logger.log('DEMO', 'cv_partition: %s, iteration: %s, MSE: %s' \
                    % (str(cv_partition+1), str(iteration+1), str(avg_test_result)))
        else:
            # logging final accuracy/error result after fine-tuning full network
            if self.CLASSIFICATION:
                self.logger.log('DEMO', 'cv_partition: %s, FINAL FULL NETWORK accuracy: %s' \
                                        % (str(cv_partition+1), str(avg_test_result)), True)
            elif self.REGRESSION:
                self.logger.log('DEMO', 'cv_partition: %s, FINAL FULL NETWORK error: %s' \
                                        % (str(cv_partition+1), str(avg_test_result)), True)

        return avg_test_result


    '''
    OVERRIDE

    log and return result - compare output to input feature vector, or handle if MLP network appended

    INPUT:
        - idx: index of test instance
        - output: network output value
        - feature_vector: actual input feature vector
        - row_vec: full row vector

    OUTPUT:
        - return error value based on difference between output/input, or handle MLP prediction layer
    '''
    def handle_prediction(self, idx, output, feature_vector, row_vec):

        self.logger.log('DEBUG', 'raw output: %s, raw actual: %s' % (str(output), str(feature_vector)))

        # if MLP network is appended
        if self.mlp_network_appended:
            actual = row_vec['CLASS']
            output = self.get_output_val(output)

            if self.CLASSIFICATION:
                self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                        % (str(idx), str(output), str(actual)) \
                                        + (' --> WRONG!' if output != actual else ''))

                # return 0/1 correctness result to list of results
                return int(output == actual)

            elif self.REGRESSION:
                self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                        % (str(idx), str(round(output, 5)), str(round(actual, 5))) \
                                        + (' --> DIFF: %s' % str(round(abs(output - actual), 5))))

                # return squared error to list of results
                return abs(output - actual)**2
        else:
            # MLP network NOT appended, we're building the SAE network, expected = input
            expected = feature_vector

            if self.CLASSIFICATION:
                self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                            % (str(idx), str(output), str(expected)))
            elif self.REGRESSION:
                self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                            % (str(idx), str(output), str(expected)))

            # return mean squared error of network output with respect to expected output
            return ((output - expected)**2).mean()


    '''
    make predictions for all test data using full SAE network with MLP on top for prediction

    INPUT:
        - cv_partition: number of cross validation partition, used for logging
        - test_data: test_data dataframe for calculating accuracy/error results

    OUTPUT:
        - return float representing accuracy/error result for given test data partition
    '''
    def predict(self, train_data, hyperparams, cv_partition, test_data):
        # append MLP network to SAE network to do classification/regression predictions
        self.append_mlp_network()
        # fine tune full network using backpropagation, SAE + MLP
        self.fine_tune_full_network(train_data, hyperparams, cv_partition, test_data)
        # make classification/regression predictions and return accuracy/error result
        return self.do_test_data(cv_partition, None, test_data)


    '''
    add prediction layer on top of SAE - regular MLP network layer for prediction
    '''
    def append_mlp_network(self):
        self.logger.log('DEMO', 'appending MLP network for prediction...')

        self.logger.log('INFO', 'FULL network shape (without MLP appended): %s' % str(self.full_network_shape))
        self.logger.log('INFO', 'weights before append_mlp_network(): %s, shape: %s' \
            % ('SKIP', str(self.utils.get_shapes(self.weights))))
        self.logger.log('INFO', 'biases before append_mlp_network(): %s, shape: %s\n' \
                        % ('SKIP', str(self.utils.get_shapes(self.biases))))

        # get deserialized MLP network to place on top of SAE network
        prediction_mlp_network = self.network_loader.load_network(self.data_set_name)

        self.logger.log('INFO', 'prediction_mlp_network.weights: %s, shape: %s' \
                        % ('SKIP', str(self.utils.get_shapes(prediction_mlp_network.weights))))
        self.logger.log('INFO', 'prediction_mlp_network.biases: %s, shape: %s\n' \
                        % ('SKIP', str(self.utils.get_shapes(prediction_mlp_network.biases))))

        mlp_layer_sizes = prediction_mlp_network.layer_sizes

        self.add_glue_weights_biases(mlp_layer_sizes[0])

        for mlp_layer_idx in range(len(mlp_layer_sizes)):
            self.logger.log('INFO', 'adding MLP layer of size: %s' % str(mlp_layer_sizes[mlp_layer_idx]))

            if mlp_layer_idx != len(mlp_layer_sizes)-1:
                # add MLP weights/biases to this network's weights/biases lists
                self.weights.append(prediction_mlp_network.weights[mlp_layer_idx])
                self.biases.append(prediction_mlp_network.biases[mlp_layer_idx])
                self.logger.log('INFO', 'after appending weights/biases for idx: %s, wshape: %s, bshape: %s' \
                    % (str(mlp_layer_idx), str(self.utils.get_shapes(self.weights)), str(self.utils.get_shapes(self.biases))))
                # add MLP activation function specs to this network's activation function specs
                self.layer_activation_funcs.append(prediction_mlp_network.layer_activation_funcs[mlp_layer_idx])

            # add MLP layer size to full network shape list
            self.full_network_shape.append(mlp_layer_sizes[mlp_layer_idx])

        self.mlp_network_appended = True
        self.logger.log('DEMO', 'FULL network shape (with MLP appended): %s' % str(self.full_network_shape))
        self.logger.log('DEMO', 'FULL network weights/biases --> wshape: %s, bshape: %s\n' \
                    % (str(self.utils.get_shapes(self.weights)), str(self.utils.get_shapes(self.biases))))


    '''
    add glue matrix and bias vector
    '''
    def add_glue_weights_biases(self, mlp_network_input_size):
        # add randomly initialized weights/biases for glue between last SAE layer and first MLP layer
        self.weights.append(np.random.randn(mlp_network_input_size, self.full_network_shape[-1]))
        self.biases.append(np.random.randn(mlp_network_input_size, 1))
        self.layer_activation_funcs.append(self.layer_activation_funcs[-1]) # add one more activation function


    '''
    fine-tune full SAE+NN network using backpropagation
    '''
    def fine_tune_full_network(self, train_data, hyperparams, cv_partition, test_data):
        '''
        for this one we need to use the train/test data to do backprop on the full SAE+NN network,
        we will use gradient descent to fine-tune the weights/biases of the entire network before testing.

        note: must unfreeze all layers before fine-tuning since we're doing backprop on full network now.
        '''
        self.logger.log('DEMO', 'fine-tuning the full network using backpropagation...\n')

        self.train_gradient_descent(train_data, hyperparams, cv_partition, test_data)


    '''
    reset auto-encoder layers for next cv partition
    '''
    def reset_ae_layers(self):
        self.logger.log('INFO', 'resetting layers to base SAE network...', True)
        # re-instantiate network with base SAE layer sizes
        sae_base_layer_sizes = infer_layer_sizes(self.data_set_name)
        self.full_network_shape = []
        self.full_network_shape.extend(sae_base_layer_sizes)
        # reset weights/biases lists
        self.weights = None
        self.biases = None
        # reset previous nabula weights/biases lists - used in momentum calculation
        self.prev_nab_weights = None
        self.prev_nab_biases = None
        # randomly re-initialize weights/biases of base network
        self.init_weights_biases()
        # reset freeze_statuses list to reflect base SAE network
        self.freeze_statuses = [True] * len(self.weights)
        self.freeze_statuses[-1] = False # output layer never frozen
        # reset MLP network appended boolean
        self.mlp_network_appended = False


    '''
    OVERRIDE

    get column vector representing expected output

    INPUT:
        - row_vec: row vector representing current data point in consideration

    OUTPUT:
        - return column vector representing expected output
            - for SAE, just return input feature vector since we're auto-encoding
    '''
    def get_expected_output(self, row_vec):

        if self.mlp_network_appended:
            return self.get_expected_output_for_mlp(row_vec)

        self.logger.log('DEBUG', 'SAE: get_expected_output: row_vec: %s, shape: %s' \
                                                % (str(row_vec), str(row_vec.shape)))
        mask = row_vec.index.isin(['CLASS'])
        feature_vector = row_vec.loc[~mask]
        expected_output = np.array(feature_vector, dtype=np.float).reshape(len(feature_vector), 1)

        self.logger.log('DEBUG', 'SAE: get_expected_output: expected_output: %s, shape: %s' \
                                        % (str(expected_output), str(expected_output.shape)))
        return expected_output


    '''
    get column vector representing expected output for MLP - i.e. [0, 0, 0, 0, 1, 0, 0, 0].transpose()

    INPUT:
        - output_val: actual output value - i.e. 'PATH' for segmentation data

    OUTPUT:
        - return column vector representing expected output
    '''
    def get_expected_output_for_mlp(self, row_vec):
        output_val = row_vec['CLASS']
        if self.REGRESSION:
            # for regression there is always a single output node
            return np.array([output_val]).reshape(1, 1)
        else:
            # CLASSIFICATION
            ordered_output_nodes = self.get_ordered_output_nodes()
            # return column vector representing expected output containing only 0/1 values
            return np.array([int(val == output_val) for val in ordered_output_nodes])\
                                                .reshape(len(ordered_output_nodes), 1)


    '''
    OVERRIDE

    compute cost of output activations given expected output

    INPUT:
        - network_output: network output - in form [0.25, 0.67, 0.1, 0.8, 0.05].transpose()
        - expected_output: expected output vector - must be in form [0, 0, 0, 1, 0].transpose()

    OUTPUT:
        - difference vector of diff between network and expected output
    '''
    def compute_cost(self, network_output, expected_output):
        self.logger.log('DEBUG', 'network_output: %s, \nexpected_output: %s' \
                                % (str(network_output), str(expected_output)))
        # the cost is simply the diff between the network output and expected output
        cost = network_output - expected_output
        self.logger.log('DEBUG', 'compute_cost: cost: %s, shape: %s' % (str(cost), str(cost.shape)))
        sum_squared_error = np.sum(np.square(cost))
        self.logger.log('DEBUG', 'SAE: compute_cost: sum_squared_error: %s' % str(sum_squared_error))
        return cost


# SCRIPT-LEVEL HELPER METHODS


'''
infer layer sizes for base SAE network (input/output) based on data set

INPUT:
    - data_set_name: name of data set

OUTPUT:
    - list representing size of input/output layers for base SAE network
'''
def infer_layer_sizes(data_set_name):
    if data_set_name == 'segmentation':
        return [19, 19]
    elif data_set_name == 'car':
        return [6, 6]
    elif data_set_name == 'abalone':
        return [8, 8]
    elif data_set_name == 'machine':
        return [9, 9]
    elif data_set_name == 'forestfires':
        return [12, 12]
    elif data_set_name == 'wine':
        return [11, 11]
    else:
        raise Exception('unknown data_set_name: %s' % str(data_set_name))



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning SAENetwork...\n')
