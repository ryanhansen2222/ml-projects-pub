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


class ResultsTests(unittest.TestCase):

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

        def test_get_likelihood_breast_cancer_data(self):
    		breast_cancer_data = self.data_api_impl.get_breast_cancer_data()
            self.assertTrue(self.lossFunctionAnalysis(breast_cancer_data, 'CLASS', [""]) == 0.32710280373831774) # class = 1 likelihood
