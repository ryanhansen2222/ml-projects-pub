#!/usr/bin/env python3


# IMPORTS


import sys
# add below directories to class path
sys.path.append('../../naive_bayes')
sys.path.append('../../data_api')

import unittest
from data_api import DataApi
from naive_bayes import NaiveBayes
import pandas as pd


# unit tests for naive_bayes script


class NaiveBayesTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')

		self.naive_bayes_impl = NaiveBayes()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# test class likelihood values for breast cancer data
	def test_get_likelihood_breast_cancer_data(self):
		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		self.assertTrue(breast_cancer_data is not None)
		self.assertTrue(self.naive_bayes_impl.get_likelihood(breast_cancer_data, 'CLASS', 2) == 0.6552217453505007) # class = 2 has 65.5% likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(breast_cancer_data, 'CLASS', 4) == 0.3447782546494993) # class = 4 has 34.5% likelihood


	# test class likelihood values for glass data
	def test_get_likelihood_glass_data(self):
		glass_data = self.data_api_impl.get_glass_data()
		self.assertTrue(glass_data is not None)
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 1) == 0.32710280373831774) # class = 1 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 2) == 0.35514018691588783) # class = 2 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 3) == 0.0794392523364486)  # class = 3 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 4) == 0.0)				 # class = 4 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 5) == 0.06074766355140187) # class = 5 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 6) == 0.04205607476635514) # class = 6 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(glass_data, 'CLASS', 7) == 0.13551401869158877) # class = 7 likelihood


	# test class likelihood values for iris data
	def test_get_likelihood_iris_data(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assertTrue(iris_data is not None)
		self.assertTrue(self.naive_bayes_impl.get_likelihood(iris_data, 'CLASS', 'Iris-setosa') == 0.3333333333333333)     # class = Iris-setosa likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(iris_data, 'CLASS', 'Iris-versicolor') == 0.3333333333333333) # class = Iris-versicolor likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(iris_data, 'CLASS', 'Iris-virginica') == 0.3333333333333333)  # class = Iris-virginica likelihood

	
	# test class likelihood values for soybean small data
	def test_get_likelihood_soybean_small_data(self):
		soybean_small_data = self.data_api_impl.get_soybean_small_data()
		self.assertTrue(soybean_small_data is not None)
		self.assertTrue(self.naive_bayes_impl.get_likelihood(soybean_small_data, 'CLASS', 'D1') == 0.2127659574468085) # class = D1 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(soybean_small_data, 'CLASS', 'D2') == 0.2127659574468085) # class = D2 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(soybean_small_data, 'CLASS', 'D3') == 0.2127659574468085) # class = D3 likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(soybean_small_data, 'CLASS', 'D4') == 0.3617021276595745) # class = D4 likelihood


	# test class likelihood values for house votes data
	def test_get_likelihood_house_votes_data(self):
		house_votes_data = self.data_api_impl.get_house_votes_data()
		self.assertTrue(house_votes_data is not None)
		self.assertTrue(self.naive_bayes_impl.get_likelihood(house_votes_data, 'CLASS', 'democrat') == 0.6137931034482759)    # class = democrat likelihood
		self.assertTrue(self.naive_bayes_impl.get_likelihood(house_votes_data, 'CLASS', 'republican') == 0.38620689655172413) # class = republican likelihood


	# test attribute value likelihoods
	def test_get_attribute_value_likelihoods(self):
		'''
		full test data frame:

			a       b   CLASS
		 0  1     cat  class1
		 1  2     cat  class2
		 2  2     dog  class2
		 3  4    wolf  class1
		 4  5  monkey  class2
		 5  5     dog  class1
		 6  5     cat  class1
		'''

		# full test data frame: 7 rows, 2 attribute columns, one numeric, one categorical
		test_data_frame = pd.DataFrame({'a':[1, 2, 2, 4, 5, 5, 5], \
			'b':['cat', 'cat', 'dog', 'wolf', 'monkey', 'dog', 'cat'], \
			'CLASS':['class1', 'class2', 'class2', 'class1', 'class2', 'class1', 'class1']})

		'''
		class1 data frame:

		  	a     b   CLASS
		 0  1   cat  class1
		 3  4  wolf  class1
		 5  5   dog  class1
		 6  5   cat  class1
		'''
		class1_data_frame = self.data_api_impl.get_class_data_frame(test_data_frame, 'class1')

		'''
		class2 data frame:

			a       b   CLASS
		 1  2     cat  class2
		 2  2     dog  class2
		 4  5  monkey  class2
		'''
		class2_data_frame = self.data_api_impl.get_class_data_frame(test_data_frame, 'class2')

		'''
		class1 likelihoods:

		'a' column should have:
			'1' --> (1 + 1) / (4 + 2) = 2/6
			'4' --> (1 + 1) / (4 + 2) = 2/6
			'5' --> (2 + 1) / (4 + 2) = 3/6
		'b' column should have:
			'cat' --> (2 + 1) / (4 + 2) = 3/6
			'wolf' --> (1 + 1) / (4 + 2) = 2/6
			'dog' --> (1 + 1) / (4 + 2) = 2/6
		'''

		class1_likelihoods = self.naive_bayes_impl.get_attribute_value_likelihoods(class1_data_frame)
		self.assertTrue(class1_likelihoods is not None)
		self.assertTrue(class1_likelihoods['a']['1'] == 2/6.0)
		self.assertTrue(class1_likelihoods['a']['4'] == 2/6.0)
		self.assertTrue(class1_likelihoods['a']['5'] == 3/6.0)
		self.assertTrue(class1_likelihoods['b']['cat'] == 3/6.0)
		self.assertTrue(class1_likelihoods['b']['wolf'] == 2/6.0)
		self.assertTrue(class1_likelihoods['b']['dog'] == 2/6.0)

		'''
		class2 likelihoods:

		'a' column should have:
			'2' --> (2 + 1) / (3 + 2) = 3/5
			'5' --> (1 + 1) / (3 + 2) = 2/5
		'b' column should have:
			'cat' --> (1 + 1) / (3 + 2) = 2/5
			'dog' --> (1 + 1) / (3 + 2) = 2/5
			'monkey' --> (1 + 1) / (3 + 2) = 2/5
		'''

		class2_likelihoods = self.naive_bayes_impl.get_attribute_value_likelihoods(class2_data_frame)
		self.assertTrue(class2_likelihoods is not None)
		self.assertTrue(class2_likelihoods['a']['2'] == 3/5.0)
		self.assertTrue(class2_likelihoods['a']['5'] == 2/5.0)
		self.assertTrue(class2_likelihoods['b']['cat'] == 2/5.0)
		self.assertTrue(class2_likelihoods['b']['dog'] == 2/5.0)
		self.assertTrue(class2_likelihoods['b']['monkey'] == 2/5.0)

		# END TEST


	# test get predicted class with iris data
	def test_get_predicted_class_iris(self):
		iris_data = self.data_api_impl.get_iris_data()
		self.assertTrue(iris_data is not None)

		# for some reason the predictions seem flipped, like some part of the calculation just needs to be inverted...
		'''
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 0))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 1))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 2))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 3))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 4))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 146))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 147))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 148))
		print(self.naive_bayes_impl.get_predicted_class(iris_data, 'iris', 149))
		'''
		


	# test get predicted class with breast_cancer data
	def test_get_predicted_class_iris(self):
		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
		self.assertTrue(breast_cancer_data is not None)

		# this seems to be working! well almost... some values are still wrong
		'''
		print(self.naive_bayes_impl.get_predicted_class(breast_cancer_data, 'breast_cancer', 0))
		print(self.naive_bayes_impl.get_predicted_class(breast_cancer_data, 'breast_cancer', 696))
		print(self.naive_bayes_impl.get_predicted_class(breast_cancer_data, 'breast_cancer', 697))
		print(self.naive_bayes_impl.get_predicted_class(breast_cancer_data, 'breast_cancer', 698))
		'''



# run the test script
if __name__ == '__main__':
    unittest.main()

