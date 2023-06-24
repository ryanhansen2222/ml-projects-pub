#!/usr/bin/env python3

import sys
# add data_api directory to class path
sys.path.append('../../data_api')

import unittest
from data_api import DataApi

class ExperimentRunnerTests(unittest.TestCase):


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


	# test run experiment
	def test_run_experiment(self):
		pass



# run the test script
if __name__ == '__main__':
    unittest.main()

