#!/usr/bin/env python3

import sys
# add data_api directory to class path
sys.path.append('../../data_api')
sys.path.append('../../preprocessing')

import unittest
from data_api import DataApi
from preprocess import Preprocess


# unit tests for preprocess script


class PreprocessTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')

		# construct preprocess instance to test
		self.preprocess_impl = Preprocess()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# test get scrambled data frame method - breast cancer data
	def test_get_scrambled_data_frame_breast_cancer_data(self):
		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		self.assert_scrambled_data_frame_not_equal(breast_cancer_data, scramble_factor=0.1)


	# test get scrambled data frame method - glass data
	def test_get_scrambled_data_frame_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assert_scrambled_data_frame_not_equal(glass_data, scramble_factor=0.1)


	# test get scrambled data frame method - iris data
	def test_get_scrambled_data_frame_iris_data(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assert_scrambled_data_frame_not_equal(iris_data, scramble_factor=0.1)


	# test get scrambled data frame method - soybean small data
	def test_get_scrambled_data_frame_soybean_small_data(self):
		soybean_small_data = self.data_api_impl.get_soybean_small_data()
		self.assert_scrambled_data_frame_not_equal(soybean_small_data, scramble_factor=0.1)


	# test get scrambled data frame method - house votes data
	def test_get_scrambled_data_frame_house_votes_data(self):
		house_votes_data = self.data_api_impl.get_house_votes_data()
		self.assert_scrambled_data_frame_not_equal(house_votes_data, scramble_factor=0.1)


	# utility method for generating scrambled data frame and asserting difference from raw data frame
	def assert_scrambled_data_frame_not_equal(self, raw_data_frame, scramble_factor):
		# scramble 10% of features in data_frame_to_scramble data frame
		scrambled_data_frame, scrambled_indexes = self.preprocess_impl.get_scrambled_data_frame(raw_data_frame, scramble_factor)
		# assert scrambled data frame is not equal to the raw data frame
		self.assertFalse(scrambled_data_frame.equals(raw_data_frame))


	# test getting a string column name from an int column index - glass data
	def test_get_column_name_from_column_index_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 0) == 'ID NUMBER')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 1) == 'RI')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 2) == 'Na')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 3) == 'Mg')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 4) == 'Al')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 5) == 'Si')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 6) == 'K')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 7) == 'Ca')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 8) == 'Ba')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 9) == 'Fe')
		self.assertTrue(self.preprocess_impl.get_column_name_from_column_index(glass_data, 10) == 'CLASS')


	# test getting an int column index from a string column name - glass data
	def test_get_column_index_from_column_name_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'ID NUMBER') == 0)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'RI') == 1)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Na') == 2)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Mg') == 3)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Al') == 4)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Si') == 5)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'K') == 6)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Ca') == 7)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Ba') == 8)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'Fe') == 9)
		self.assertTrue(self.preprocess_impl.get_column_index_from_column_name(glass_data, 'CLASS') == 10)


	# test preprocess breast cancer data - remove rows with missing values
	def test_preprocess_breast_cancer_data(self):
		raw_breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		preprocessed_breast_cancer_data = self.preprocess_impl.preprocess_raw_data_frame(raw_breast_cancer_data, 'breast_cancer')
		for row_index in range(preprocessed_breast_cancer_data.shape[0]):
			row_data = preprocessed_breast_cancer_data.iloc[row_index, :].values
			self.assertTrue(len(row_data[row_data == '?']) == 0) # verify no '?' values remaining in preprocessed data frame
		self.assertTrue(preprocessed_breast_cancer_data.shape[0] == 683) # removed 16 rows from breast cancer data (rows with missing values)
		self.assertTrue(preprocessed_breast_cancer_data.shape[1] == 10) # removed the 'ID NUMBER' column


	# test preprocess glass data - binning into 5 bins
	def test_preprocess_glass_data(self):
		'''
		Summary Statistics:
			Attribute:   Min     Max      Mean     SD      Correlation with class
			 2. RI:       1.5112  1.5339   1.5184  0.0030  -0.1642
			 3. Na:      10.73   17.38    13.4079  0.8166   0.5030
			 4. Mg:       0       4.49     2.6845  1.4424  -0.7447
			 5. Al:       0.29    3.5      1.4449  0.4993   0.5988
			 6. Si:      69.81   75.41    72.6509  0.7745   0.1515
			 7. K:        0       6.21     0.4971  0.6522  -0.0100
			 8. Ca:       5.43   16.19     8.9570  1.4232   0.0007
			 9. Ba:       0       3.15     0.1750  0.4972   0.5751
			10. Fe:       0       0.51     0.0570  0.0974  -0.1879
		'''
		raw_glass_data = self.data_api_impl.get_glass_data()
		preprocessed_glass_data = self.preprocess_impl.preprocess_raw_data_frame(raw_glass_data, 'glass')
		self.assertTrue(preprocessed_glass_data is not None)
		self.assertTrue(preprocessed_glass_data.shape[0] == 214) # same number of rows as raw glass data frame
		self.assertTrue(preprocessed_glass_data.shape[1] == 10) # removed the 'ID NUMBER' column

		#print(preprocessed_glass_data)

		# verify each row value is between 1 and 5 inclusive (5 bins)
		#self.verify_bin_values(preprocessed_glass_data, 1, 5)


	# test preprocess iris data - binning into 5 bins
	def test_preprocess_iris_data(self):
		'''
		Summary Statistics:

			Min  Max   Mean    SD   Class Correlation

		    sepal length: 4.3  7.9   5.84  0.83    0.7826   
		    sepal width: 2.0  4.4   3.05  0.43   -0.4194
		    petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
		    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
		'''
    	
		raw_iris_data = self.data_api_impl.get_iris_data()
		preprocessed_iris_data = self.preprocess_impl.preprocess_raw_data_frame(raw_iris_data, 'iris')
		self.assertTrue(preprocessed_iris_data is not None)
		self.assertTrue(preprocessed_iris_data.shape[0] == 150) # same number of rows as raw iris data frame
		self.assertTrue(preprocessed_iris_data.shape[1] == 5) # same number of columns as raw iris data frame

		#print(preprocessed_iris_data)

		# verify each row value is between 1 and 5 inclusive (5 bins)
		#self.verify_bin_values(preprocessed_iris_data, 1, 5)


	# utility method: verify every bin value in data frame
	def verify_bin_values(self, data_frame, min_bin_val, max_bin_val):
		for row_index in range(data_frame.shape[0]):
			row_data = data_frame.iloc[row_index, :]
			if row_data.name != 'CLASS':
				for row_val in row_data.values:
					if row_val is str:
						continue
					if row_val < min_bin_val or row_val > max_bin_val:
						print('ERROR => ' + str(row_val))
					self.assertTrue(row_val >= min_bin_val)
					self.assertTrue(row_val <= max_bin_val)


	# test removing '?' values from house_votes data frame
	def test_preprocess_house_votes_data(self):
		raw_house_votes_data = self.data_api_impl.get_house_votes_data()
		preprocessed_house_votes_data = self.preprocess_impl.preprocess_raw_data_frame(raw_house_votes_data, 'house_votes')
		for row_index in range(preprocessed_house_votes_data.shape[0]):
			row_data = preprocessed_house_votes_data.loc[row_index, :].values
			self.assertTrue(len(row_data[row_data == '?']) == 0) # verify no '?' values remaining in preprocessed data frame
		self.assertTrue(preprocessed_house_votes_data.shape[0] == 435) # same number of rows as raw house votes data frame
		self.assertTrue(preprocessed_house_votes_data.shape[1] == 17) # same number of columns as raw house votes data frame


# EXECUTE SCRIPT


# run the test script
if __name__ == '__main__':
    unittest.main()

