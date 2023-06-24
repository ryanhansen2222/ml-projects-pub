#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../data')
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../cross_validation')
sys.path.append('../naive_bayes')
sys.path.append('../results_processor')

import pandas as pd
from data_api import DataApi
from preprocess import Preprocess
from cross_validation import CrossValidation
from naive_bayes import NaiveBayes
from results_processor import ResultsProcessor


# CLASS


'''
	This class is responsible for running all the different parts of the application. 
	It uses an instance of the DataApi for fetching the data from the csv files, and 
	then passes that data to the Preprocessor for preprocessing and possibly scrambling.

	Then it performs 10-fold cross validation on the data using the CrossValidation class, 
	and runs the Naive Bayes algorithm on each combination of training/test sets using the 
	NaiveBayes class. The intermediate loss function results are calculated using the 
	ResultsProcessor, and the run_experiment() method below returns the average loss 
	function results using zero one loss and mean squared error as error metrics.
	
	The entire set of experiments is kicked off by simply running this script.
'''


class ExperimentRunner():


	def __init__(self):
		self.DEBUG = False

		# get instances of all the classes needed to run an experiment
		self.data_api_impl = DataApi('../../data/')
		self.preprocess_impl = Preprocess()
		self.cross_validation_impl = CrossValidation()
		self.naive_bayes_impl = NaiveBayes()
		self.results_processor_impl = ResultsProcessor()


	# run experiment - i.e. run naive bayes on the data set with data_frame_name, and possibly scramble
	def run_experiment(self, data_frame_name, scramble):

		print('\ndata_frame_name => ' + data_frame_name)

		# get raw data frame to run experiment against
		raw_data_frame = self.data_api_impl.get_full_data_frame(data_frame_name)

		print('\nraw_data_frame:\n')
		print(raw_data_frame)

		# preprocess data
		preprocessed_data_frame = self.preprocess_impl.preprocess_raw_data_frame(raw_data_frame, data_frame_name)

		print('\npreprocessed_data_frame:\n')
		print(preprocessed_data_frame)

		# scramble data if we want to introduce noise
		if scramble == True:
			print('\nSCRAMBLING DATA...')
			# scramble 10% of features and return scrambled data frame into preprocessed_data_frame variable
			preprocessed_data_frame, column_indexes_scrambled = self.preprocess_impl.get_scrambled_data_frame(preprocessed_data_frame, 0.1)

		# get indexes list for data frame cross validation - a list of row numbers used to partition the data
		data_frame_indexes_list = self.cross_validation_impl.get_indexes_list(preprocessed_data_frame)

		print('\ndata_frame_indexes_list for cross validation:\n')
		print(data_frame_indexes_list)

		# nested dictionary to hold naive bayes performance results for each combination of training/test sets
		# key pattern --> key = test_set_1 , where the number at the end of the key is the test set index
		# each value is another dictionary with keys = { 'zero_one_loss', 'mean_squared_error' }
		# the nested dictionary values are the corresponding loss function metrics for predictions using the test set
		cross_validation_results = {}

		# list of sizes of test sets used for getting average test set size
		test_set_sizes = []

		# for each test set used in 10-fold cross validation
		for test_set_index in range(1, 11):

			# initialize key and corresponding nested dictionary in results dictionary
			test_set_key = 'test_set_' + str(test_set_index)
			cross_validation_results[test_set_key] = {}

			# get training set (full data frame - rows in test_set_index bucket)
			training_set = self.cross_validation_impl.get_training_set(preprocessed_data_frame, test_set_index, data_frame_indexes_list)

			# get test set (rows in test_set_index bucket)
			test_set = self.cross_validation_impl.get_test_set(preprocessed_data_frame, test_set_index, data_frame_indexes_list)

			test_set_sizes.append(test_set.shape[0]) # add number of rows in test set to test_set_sizes list

			# run naive bayes on training set / test set combination
			# returns dictionary where key is the row index (as string) and value is the predicted class for that row
			prediction_results = self.naive_bayes_impl.run_naive_bayes(training_set, test_set)

			# calculate loss function results given prediction results - measure prediction accuracy
			zero_one_loss, mean_squared_error = self.results_processor_impl.loss_function_analysis(raw_data_frame, data_frame_name, prediction_results)
			
			cross_validation_results[test_set_key]['zero_one_loss'] = zero_one_loss
			cross_validation_results[test_set_key]['mean_squared_error'] = mean_squared_error

			#print(test_set_key + ' => ( zero_one_loss: ' + str(zero_one_loss) + ', mean_squared_error: ' + str(mean_squared_error) + ' )')

		# calculate average loss function results over all cross validation folds
		avg_zero_one_loss, avg_mean_squared_error = self.results_processor_impl.get_avg_loss_vals(cross_validation_results)
		avg_test_set_size = sum(test_set_sizes) / len(test_set_sizes) # get average test set size for reference

		print('\nAVERAGE RESULTS: (test_set_size: ' + str(avg_test_set_size) + ') => ' \
			+ '(zero_one_loss: ' + str(avg_zero_one_loss) + ', mean_squared_error: ' + str(avg_mean_squared_error) + ')')

		print('---------------------------------------------------------------------------------------------------------------------')

		# return tuple with average loss function results
		return (avg_zero_one_loss, avg_mean_squared_error)



# EXECUTE SCRIPT


# if we run this script directly, the following code below will execute
if __name__ == "__main__":

	print('\nrunning experiments...')
	
	# get instance of experiment runner class
	experiment_runner_impl = ExperimentRunner()

	# list of data sets to run experiments on
	data_sets_names = ['breast_cancer', 'glass', 'iris', 'soybean_small', 'house_votes']

	# dictionaries for storing results
	regular_data_sets_results = {}
	scrambled_data_sets_results = {}

	print('-------------------------------------------------------------------------------------------------------------------')

	# regular data sets

	print('\n\n------------------------------------------------ REGULAR DATA SETS ------------------------------------------------\n')

	# run experiment on each data set, without scrambling
	for data_set in data_sets_names:
		regular_data_sets_results[data_set] = experiment_runner_impl.run_experiment(data_set, scramble=False)

	# scrambled data sets

	print('\n\n------------------------------------------------ SCRAMBLED DATA SETS ------------------------------------------------\n')

	# run experiment on each data set, with scrambling
	for data_set in data_sets_names:
		scrambled_data_sets_results[data_set] = experiment_runner_impl.run_experiment(data_set, scramble=True)


	# print results

	print('\n\n---------------------------------------------------- COMPARISON -----------------------------------------------------\n\n')

	print('zero_one_loss:')

	print('\tregular:')

	for data_set in data_sets_names:
		regular_results = regular_data_sets_results[data_set]
		print('\t\t' + data_set + ' ---> ' + str(regular_results[0]))

	print('\tscrambled:')

	for data_set in data_sets_names:
		scrambled_results = scrambled_data_sets_results[data_set]
		print('\t\t' + data_set + ' ---> ' + str(scrambled_results[0]))

	print('\nmean_squared_error:')

	print('\tregular:')

	for data_set in data_sets_names:
		regular_results = regular_data_sets_results[data_set]
		print('\t\t' + data_set + ' ---> ' + str(regular_results[1]))

	print('\tscrambled:')

	for data_set in data_sets_names:
		scrambled_results = scrambled_data_sets_results[data_set]
		print('\t\t' + data_set + ' ---> ' + str(scrambled_results[1]))


	print('\n---------------------------------------------------------------------------------------------------------------------')

	print('\nDONE')


	# DONE


