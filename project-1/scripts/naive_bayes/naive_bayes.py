#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')

import pandas as pd
from data_api import DataApi


# CLASS


'''
	This class is responsible for running everything related to the Naive Bayes algorithm. 
	It has methods for getting the likelihood of a class value, as well as the likelihood 
	of each unique attribute value in every column. The most important method in the class 
	is get_predicted_class_using_training_data(), which returns a predicted class given a 
	training set and a test row.

	The run_naive_bayes() method runs one execution of the Naive Bayes algorithm given a 
	training set and a test set, and returns a dictionary of predictions that can be 
	processed by the ResultsProcessor class to determine the corresponding average prediction accuracy.
'''


class NaiveBayes:


	def __init__(self):
		self.DEBUG = False

		# instantiate data_api instance for access to data
		self.data_api_impl = DataApi('../../../data/')


	# get likelihood of class value for a given data frame
	def get_likelihood(self, data_frame, column_name, class_value):
		counter = 0
		column_data = data_frame[column_name]
		# iterate through column and get count of all rows with class_value
		for value in column_data:
			if value == class_value:
				counter = counter + 1
		# return ratio: number of rows with class_value divided by total number of rows
		return counter / len(column_data)


	# iterate over all columns and calculate attribute value likelihoods
	def get_attribute_value_likelihoods(self, class_data_frame):
		# return nested dictionary with likelihood values for each column/value combination
		likelihoods = {}
		column_names = list(class_data_frame.columns.values)

		for column in column_names:
			# skip CLASS column since we already know every row has the same class value
			if column == 'CLASS' or column == 'ID NUMBER':
				continue

			# each column index stores another dictionary consisting of (unique value => count) mappings
			likelihoods[column] = {}
			column_data = class_data_frame[column]
			column_count = len(column_data)
			unique_attr_vals = column_data.unique() # get list of unique values in column_data list
			attr_val_likelihoods = {} # inner dictionary for likelihood of each unique value in column

			# initialize all inner dictionary values to None
			for unique_attr_val in unique_attr_vals:
				attr_val_likelihoods[str(unique_attr_val)] = None

			'''
			from project description:

				"for each attribute value, divide the number of examples that match that attribute
				 value (plus one) by the number of examples in the class (plus number of attributes)"
			'''

			# get attribute value likelihood for each unique value in column, store in nested dictionary
			for unique_attr_val in unique_attr_vals:
				attr_val_likelihood = (self.get_unique_attr_val_count(column_data, unique_attr_val) + 1) \
										/ (column_count + len(column_names) - 1) # -1 for ignoring CLASS column
				attr_val_likelihoods[str(unique_attr_val)] = attr_val_likelihood

			# update top-level dictionary column key with attribute value likelihoods
			likelihoods[column] = attr_val_likelihoods

		# return nested dictionary of attribute value likelihoods for each unique attribute value
		return likelihoods


	# get count of occurrences of unique attribute value in given column
	def get_unique_attr_val_count(self, column_data, unique_attr_val):
		counter = 0
		for attr_val in column_data:
			if attr_val == unique_attr_val:
				counter = counter + 1
		return counter


	# get predicted class for given instance at instance_index of given data_frame
	def get_predicted_class(self, data_frame, data_frame_name, instance_index):
		class_labels = data_frame['CLASS'].unique() # get list of unique class values
		class_probs = {} # store class probabilities in dictionary

		# for each unique class value, calculate probability of the test instance being in the class
		for class_label in class_labels:
			# get likelihood of class value
			class_likelihood = self.get_likelihood(data_frame, 'CLASS', class_label)
			# get subset data frame of all rows with given class value
			class_data_frame = self.data_api_impl.get_data_frame_for_class(data_frame_name, class_label)
			# calculate likelihoods for each unique attribute value
			attr_val_likelihoods = self.get_attribute_value_likelihoods(class_data_frame)
			# get instance data for test row at given instance_index of data_frame
			instance_data = data_frame.loc[instance_index, :]
			# calculate probability product for all attribute likelihoods
			prob_product = self.get_prob_product(attr_val_likelihoods, instance_data)
			# store class probability in dictionary
			class_probs[class_label] = class_likelihood * prob_product

		# return class with maximum probability for given test row
		return self.get_class_max_prob(class_probs)


	# get predicted class for given test_data_row given a training data set
	def get_predicted_class_using_training_data(self, training_data_frame, test_data_row):
		class_labels = training_data_frame['CLASS'].unique() # get list of unique class values
		class_probs = {} # store class probabilities in dictionary

		# for each unique class value, calculate probability of the test instance being in the class
		for class_label in class_labels:
			# get likelihood of class value
			class_likelihood = self.get_likelihood(training_data_frame, 'CLASS', class_label)
			# get subset data frame of all rows with given class value
			class_data_frame = self.data_api_impl.get_all_class_instances(training_data_frame, class_label)
			# calculate likelihoods for each unique attribute value
			attr_val_likelihoods = self.get_attribute_value_likelihoods(class_data_frame)
			# calculate probability product for all attribute likelihoods
			prob_product = self.get_prob_product(attr_val_likelihoods, test_data_row)
			# store class probability in dictionary
			class_probs[class_label] = class_likelihood * prob_product

		# return class with maximum probability for given test_data_row
		return self.get_class_max_prob(class_probs)


	# get product of attribute value likelihoods for all attributes
	def get_prob_product(self, attr_val_likelihoods, instance):
		prob_product = 1
		for column in attr_val_likelihoods:
			if str(instance[column]) in attr_val_likelihoods[column]:
				prob_product = prob_product * attr_val_likelihoods[column][str(instance[column])]
		return prob_product


	# utility method for returning class label for class with highest probability
	def get_class_max_prob(self, class_probs):
		max_prob_val = 0
		max_prob_class = None
		for class_label in class_probs:
			if class_probs[class_label] > max_prob_val:
				max_prob_val = class_probs[class_label]
				max_prob_class = class_label
		# return class label for class with maximum probability
		return max_prob_class


	# run full naive bayes algorithm using training set and test set
	def run_naive_bayes(self, training_set, test_set):

		# return dictionary where key is the row index (as string) and value is the predicted class for that row
		predictions = {}

		# predict class for each instance in test set
		for test_set_index in range(test_set.shape[0]):
			test_instance = test_set.iloc[test_set_index, :] # get row data for test instance
			prediction = self.get_predicted_class_using_training_data(training_set, test_instance)
			predictions[str(test_instance.name)] = prediction

		#print(predictions)
		return predictions
			


if __name__ == "__main__":

	print('running naive_bayes...')

