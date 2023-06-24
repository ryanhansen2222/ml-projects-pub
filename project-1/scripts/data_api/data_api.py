#!/usr/bin/env python3


# IMPORTS


import sys
# add data directory to class path
sys.path.append('../../data')

import pandas as pd


# CLASS


'''
	This class is responsible for all things related to actually getting the data from the csv .data files. 
	It has various utility methods for getting specific rows/columns/values from the data sets.
	We use the "pandas" library as our data structure for holding the data matrices themselves.
'''


class DataApi():


	def __init__(self, relative_path_prefix):
		self.DEBUG = False
		# the relative path prefix is used to properly instantiate the class from anywhere in the application
		# the get_data_frame method below uses it to correctly locate the data set file by relative file path
		self.relative_path_prefix = relative_path_prefix


	# utility method: read csv file at file name arg and return data matrix as pandas data frame
	def get_data_frame(self, data_set, column_names):
		data_set_file_name = self.relative_path_prefix + data_set
		return pd.read_csv(data_set_file_name, names=column_names)


	# get pandas data frame consisting of all rows from data_set_name with given class value
	def get_data_frame_for_class(self, data_set_name, class_value):
		full_data_frame = pd.DataFrame()

		if data_set_name == 'breast_cancer':
			full_data_frame = self.get_breast_cancer_data()
		elif data_set_name == 'glass':
			full_data_frame = self.get_glass_data()
		elif data_set_name == 'iris':
			full_data_frame = self.get_iris_data()
		elif data_set_name == 'soybean_small':
			full_data_frame = self.get_soybean_small_data()
		elif data_set_name == 'house_votes':
			full_data_frame = self.get_house_votes_data()
		else:
			print('ERROR: Unknown data_set_name => ' + data_set_name)

		# set match var to all indexes that have the given class value
		match = full_data_frame['CLASS'] == class_value
		return full_data_frame[match]


	# return data frame of all rows with given class value
	def get_all_class_instances(self, data_frame, class_value):
		match = data_frame['CLASS'] == class_value
		return data_frame[match]


	# generic method for getting class data frame from full data frame
	def get_class_data_frame(self, full_data_frame, class_value):
		match = full_data_frame['CLASS'] == class_value
		return full_data_frame[match]


	# get breast cancer data as pandas data frame - specify column labels
	def get_breast_cancer_data(self):
		breast_cancer_labels = ['ID NUMBER', 'CT', 'UC_size', 'UC_shape', 'MA', 'SECS', 'BN', 'BC', 'NN', 'M', 'CLASS']
		return self.get_data_frame('breast-cancer-wisconsin.data', breast_cancer_labels)


	# get glass data as pandas data frame - specify column labels
	def get_glass_data(self):
		glass_labels = ['ID NUMBER', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'CLASS']
		return self.get_data_frame('glass.data', glass_labels)


	# get iris data as pandas data frame - specify column labels
	def get_iris_data(self):
		iris_labels = ['sepal length', 'sepal width', 'petal length', 'petal width', 'CLASS']
		return self.get_data_frame('iris.data', iris_labels)


	# get soybean small data as pandas data frame - specify column labels
	def get_soybean_small_data(self):
		# this is a messy way to specify the column labels but it'll work for now at least...
		soybean_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
						  '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
						  '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', 'CLASS']
		return self.get_data_frame('soybean-small.data', soybean_labels)


	# get house votes data as pandas data frame - specify column labels
	def get_house_votes_data(self):
		# again, this labeling is messy but oh well, not a big deal right now
		house_votes_labels = ['CLASS', '2', '3', '4', '5', '6', '7', '8',
							  '9', '10', '11', '12', '13', '14', '15', '16', '17']
		return self.get_data_frame('house-votes-84.data', house_votes_labels)


	# UTILITY METHODS


	# method for getting full data frame given data set name
	def get_full_data_frame(self, data_set_name):
		full_data_frame = pd.DataFrame()

		if data_set_name == 'breast_cancer':
			full_data_frame = self.get_breast_cancer_data()
		elif data_set_name == 'glass':
			full_data_frame = self.get_glass_data()
		elif data_set_name == 'iris':
			full_data_frame = self.get_iris_data()
		elif data_set_name == 'soybean_small':
			full_data_frame = self.get_soybean_small_data()
		elif data_set_name == 'house_votes':
			full_data_frame = self.get_house_votes_data()
		else:
			print('ERROR: Unknown data_set_name => ' + data_set_name)

		return full_data_frame


	# simple method for getting the class value of a given row in a data set
	def get_class_value_for_data_frame_row(self, data_frame, row_index):
		return data_frame.loc[row_index, 'CLASS']


	# method for getting the class value for a given row given just the data set name
	def get_class_value_for_row(self, data_frame_name, row_index):
		full_data_frame = self.get_full_data_frame(data_frame_name)
		return full_data_frame.loc[row_index, 'CLASS']


# EXECUTE SCRIPT


if __name__ == "__main__":

	print('running data_api...')
	
