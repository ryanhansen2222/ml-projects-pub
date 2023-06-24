#!/usr/bin/env python3

# !!! NOTE: this script has to been ran from the data_api root directory on command line !!!


# IMPORTS


import sys
# add data_api directory to class path
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi


# CLASS


'''
	Simple utility class for viewing data sets from command line.
'''


class ViewData():


	def __init__(self):
		self.DEBUG = False
		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../data/')


	def view_breast_cancer_data(self):
		print('\nBREAST CANCER DATA:\n')
		print(self.data_api_impl.get_breast_cancer_data())


	def view_glass_data(self):
		print('\nGLASS DATA:\n')
		print(self.data_api_impl.get_glass_data())


	def view_iris_data(self):
		print('\nIRIS DATA:\n')
		print(self.data_api_impl.get_iris_data())


	def view_soybean_small_data(self):
		print('\nSOYBEAN SMALL DATA:\n')
		print(self.data_api_impl.get_soybean_small_data())


	def view_house_votes_data(self):
		print('\nHOUSE VOTES DATA:\n')
		print(self.data_api_impl.get_house_votes_data())


# EXECUTE SCRIPT


# run the view_data script to view data matrices from command line for easy reference
if __name__ == '__main__':

	view_data_impl = ViewData()

	view_data_impl.view_breast_cancer_data()
	view_data_impl.view_glass_data()
	view_data_impl.view_iris_data()
	view_data_impl.view_soybean_small_data()
	view_data_impl.view_house_votes_data()

