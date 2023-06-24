#!/usr/bin/env python3

import sys
# add data_api directory to class path
sys.path.append('../../data_api')

import unittest
from data_api import DataApi

class DataApiTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# full data frames

	# test breast cancer data retrieval, number of rows/columns
	def test_get_breast_cancer_data(self):
		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		self.assertTrue(breast_cancer_data is not None)
		self.assertTrue(breast_cancer_data.shape[0] == 699) # 699 rows in breast cancer data matrix
		self.assertTrue(breast_cancer_data.shape[1] == 11)  # 11 attribute columns in breast cancer data matrix


	# test glass data retrieval, number of rows/columns
	def test_get_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assertTrue(glass_data is not None)
		self.assertTrue(glass_data.shape[0] == 214) # 214 rows in glass data matrix
		self.assertTrue(glass_data.shape[1] == 11)  # 11 attribute columns in glass data matrix


	# test iris data retrieval, number of rows/columns
	def test_get_iris_data(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assertTrue(iris_data is not None)
		self.assertTrue(iris_data.shape[0] == 150) # 150 rows in iris data matrix
		self.assertTrue(iris_data.shape[1] == 5)   # 5 attribute columns in iris data matrix


	# test soybean small data retrieval, number of rows/columns
	def test_get_soybean_small_data(self):
		soybean_small_data = self.data_api_impl.get_soybean_small_data()
		self.assertTrue(soybean_small_data is not None)
		self.assertTrue(soybean_small_data.shape[0] == 47) # 47 rows in soybean small data matrix
		self.assertTrue(soybean_small_data.shape[1] == 36) # 36 attribute columns in soybean small data matrix


	# test house votes data retrieval, number of rows/columns
	def test_get_house_votes_data(self):
		house_votes_data = self.data_api_impl.get_house_votes_data()
		self.assertTrue(house_votes_data is not None)
		self.assertTrue(house_votes_data.shape[0] == 435) # 435 rows in house votes data matrix
		self.assertTrue(house_votes_data.shape[1] == 17)  # 17 attribute columns in house votes data matrix


	# class data frames

	# test class frames for breast cancer data
	def test_get_data_frame_for_class_breast_cancer(self):
		class_frame_2 = self.data_api_impl.get_data_frame_for_class('breast_cancer', 2)
		self.assertTrue(class_frame_2.shape[0] == 458) # 458 rows that have class 2
		self.assertTrue(class_frame_2.shape[1] == 11)  # 11 columns, same as full data frame
		class_frame_4 = self.data_api_impl.get_data_frame_for_class('breast_cancer', 4)
		self.assertTrue(class_frame_4.shape[0] == 241) # 241 rows that have class 4
		self.assertTrue(class_frame_4.shape[1] == 11)  # 11 columns, same as full data frame


	# test class frames for glass data
	def test_get_data_frame_for_class_glass(self):
		class_frame_1 = self.data_api_impl.get_data_frame_for_class('glass', 1)
		self.assertTrue(class_frame_1.shape[0] == 70) # 70 rows that have class 1
		self.assertTrue(class_frame_1.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_2 = self.data_api_impl.get_data_frame_for_class('glass', 2)
		self.assertTrue(class_frame_2.shape[0] == 76) # 76 rows that have class 2
		self.assertTrue(class_frame_2.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_3 = self.data_api_impl.get_data_frame_for_class('glass', 3)
		self.assertTrue(class_frame_3.shape[0] == 17) # 17 rows that have class 3
		self.assertTrue(class_frame_3.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_4 = self.data_api_impl.get_data_frame_for_class('glass', 4)
		self.assertTrue(class_frame_4.shape[0] == 0)  # 0 rows that have class 4
		self.assertTrue(class_frame_4.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_5 = self.data_api_impl.get_data_frame_for_class('glass', 5)
		self.assertTrue(class_frame_5.shape[0] == 13) # 13 rows that have class 5
		self.assertTrue(class_frame_5.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_6 = self.data_api_impl.get_data_frame_for_class('glass', 6)
		self.assertTrue(class_frame_6.shape[0] == 9)  # 9 rows that have class 6
		self.assertTrue(class_frame_6.shape[1] == 11) # 11 columns, same as full data frame

		class_frame_7 = self.data_api_impl.get_data_frame_for_class('glass', 7)
		self.assertTrue(class_frame_7.shape[0] == 29) # 29 rows that have class 7
		self.assertTrue(class_frame_7.shape[1] == 11) # 11 columns, same as full data frame


	# test class frames for iris data
	def test_get_data_frame_for_class_iris(self):
		class_frame_iris_setosa = self.data_api_impl.get_data_frame_for_class('iris', 'Iris-setosa')
		self.assertTrue(class_frame_iris_setosa.shape[0] == 50) # 50 rows that have class Iris-setosa
		self.assertTrue(class_frame_iris_setosa.shape[1] == 5)  # 5 columns, same as full data frame

		class_frame_iris_versicolor = self.data_api_impl.get_data_frame_for_class('iris', 'Iris-versicolor')
		self.assertTrue(class_frame_iris_versicolor.shape[0] == 50) # 50 rows that have class Iris-versicolor
		self.assertTrue(class_frame_iris_versicolor.shape[1] == 5)  # 5 columns, same as full data frame

		class_frame_iris_virginica = self.data_api_impl.get_data_frame_for_class('iris', 'Iris-virginica')
		self.assertTrue(class_frame_iris_virginica.shape[0] == 50) # 50 rows that have class Iris-virginica
		self.assertTrue(class_frame_iris_virginica.shape[1] == 5)  # 5 columns, same as full data frame


	# test class frames for soybean small data
	def test_get_data_frame_for_class_soybean_small(self):
		class_frame_D1 = self.data_api_impl.get_data_frame_for_class('soybean_small', 'D1')
		self.assertTrue(class_frame_D1.shape[0] == 10) # 10 rows that have class D1
		self.assertTrue(class_frame_D1.shape[1] == 36) # 36 columns, same as full data frame

		class_frame_D2 = self.data_api_impl.get_data_frame_for_class('soybean_small', 'D2')
		self.assertTrue(class_frame_D2.shape[0] == 10) # 10 rows that have class D2
		self.assertTrue(class_frame_D2.shape[1] == 36) # 36 columns, same as full data frame

		class_frame_D3 = self.data_api_impl.get_data_frame_for_class('soybean_small', 'D3')
		self.assertTrue(class_frame_D3.shape[0] == 10) # 10 rows that have class D3
		self.assertTrue(class_frame_D3.shape[1] == 36) # 36 columns, same as full data frame

		class_frame_D4 = self.data_api_impl.get_data_frame_for_class('soybean_small', 'D4')
		self.assertTrue(class_frame_D4.shape[0] == 17) # 17 rows that have class D4
		self.assertTrue(class_frame_D4.shape[1] == 36) # 36 columns, same as full data frame


	# test class frames for house votes data
	def test_get_data_frame_for_class_house_votes(self):
		class_frame_democrat = self.data_api_impl.get_data_frame_for_class('house_votes', 'democrat')
		self.assertTrue(class_frame_democrat.shape[0] == 267) # 267 rows that have class democrat
		self.assertTrue(class_frame_democrat.shape[1] == 17)  # 17 columns, same as full data frame

		class_frame_republican = self.data_api_impl.get_data_frame_for_class('house_votes', 'republican')
		self.assertTrue(class_frame_republican.shape[0] == 168) # 168 rows that have class republican
		self.assertTrue(class_frame_republican.shape[1] == 17)  # 17 columns, same as full data frame


	# tests for utility methods


	def test_get_class_value_for_row(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assertTrue(self.data_api_impl.get_class_value_for_row('iris', 0) == 'Iris-setosa')
		self.assertTrue(self.data_api_impl.get_class_value_for_row('iris', 50) == 'Iris-versicolor')
		self.assertTrue(self.data_api_impl.get_class_value_for_row('iris', 100) == 'Iris-virginica')


# run the test script
if __name__ == '__main__':
    unittest.main()

