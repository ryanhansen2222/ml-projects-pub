#!/usr/bin/env python3


# IMPORTS


import sys
# add below directories to class path
sys.path.append('../../data_api')
sys.path.append('../../cross_validation')

import unittest
import pandas as pd
from data_api import DataApi
from cross_validation import CrossValidation


# unit tests for cross_validation script


class CrossValidationTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		self.data_api_impl = DataApi('../../../data/')
		self.cross_validation_impl = CrossValidation()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# test get indexes list for breast cancer data
	def test_get_indexes_list_breast_cancer_data(self):
		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		self.assertTrue(breast_cancer_data is not None)
		breast_cancer_indexes = self.cross_validation_impl.get_indexes_list(breast_cancer_data)
		self.assertTrue(len(breast_cancer_indexes) == 699) # 699 rows in breast cancer data frame
		for i in range(1, 10):
			self.assertTrue(breast_cancer_indexes.count(i) == 69) # each subset has 69 rows
		self.assertTrue(breast_cancer_indexes.count(10) == 78) # last subset has 69 + remaining...


	# test get indexes list for glass data
	def test_get_indexes_list_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assertTrue(glass_data is not None)
		glass_indexes = self.cross_validation_impl.get_indexes_list(glass_data)
		self.assertTrue(len(glass_indexes) == 214) # 214 rows in glass data frame
		for i in range(1, 10):
			self.assertTrue(glass_indexes.count(i) == 21) # each subset has 21 rows
		self.assertTrue(glass_indexes.count(10) == 25) # last subset has 21 + remaining...


	# test get indexes list for iris data
	def test_get_indexes_list_iris_data(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assertTrue(iris_data is not None)
		iris_indexes = self.cross_validation_impl.get_indexes_list(iris_data)
		self.assertTrue(len(iris_indexes) == 150) # 150 rows in iris data frame
		for i in range(1, 11):
			self.assertTrue(iris_indexes.count(i) == 15) # each subset has 15 rows


	# test get indexes list for soybean small data
	def test_get_indexes_list_soybean_small_data(self):
		soybean_small_data = self.data_api_impl.get_soybean_small_data()
		self.assertTrue(soybean_small_data is not None)
		soybean_small_indexes = self.cross_validation_impl.get_indexes_list(soybean_small_data)
		self.assertTrue(len(soybean_small_indexes) == 47) # 47 rows in soybean small data frame
		for i in range(1, 10):
			self.assertTrue(soybean_small_indexes.count(i) == 4) # each subset has 4 rows
		self.assertTrue(soybean_small_indexes.count(10) == 11) # hmmm this might be a problem...


	# test get indexes list for house votes data
	def test_get_indexes_list_house_votes_data(self):
		house_votes_data = self.data_api_impl.get_house_votes_data()
		self.assertTrue(house_votes_data is not None)
		house_votes_indexes = self.cross_validation_impl.get_indexes_list(house_votes_data)
		self.assertTrue(len(house_votes_indexes) == 435) # 435 rows in house votes data frame
		for i in range(1, 10):
			self.assertTrue(house_votes_indexes.count(i) == 43) # each subset has 43 rows
		self.assertTrue(house_votes_indexes.count(10) == 48) # hmmm...


	# TRAINING SET


	# test get training set 2 with glass data
	def test_get_training_set(self):
		glass_data = self.data_api_impl.get_glass_data()
		glass_data_indexes_list = self.cross_validation_impl.get_indexes_list(glass_data)
		glass_data_training_set = self.cross_validation_impl.get_training_set(glass_data, 2, glass_data_indexes_list)
		self.assertTrue(glass_data_training_set.shape[0] == 193) # 214 - 21 rows in test set 2 = 193 rows in training set
		self.assertTrue(glass_data_training_set.shape[1] == 11) # number of columns does not change


	# TEST SET


	# test get test set (-2) with glass data
	def test_get_test_set(self):
		glass_data = self.data_api_impl.get_glass_data()
		glass_data_indexes_list = self.cross_validation_impl.get_indexes_list(glass_data)
		glass_data_test_set = self.cross_validation_impl.get_test_set(glass_data, 2, glass_data_indexes_list)
		self.assertTrue(glass_data_test_set.shape[0] == 21) # 21 rows in test set 2
		self.assertTrue(glass_data_test_set.shape[1] == 11) # number of columns does not change



# run the test script
if __name__ == '__main__':
    unittest.main()

