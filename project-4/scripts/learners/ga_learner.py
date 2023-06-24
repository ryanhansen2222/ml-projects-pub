#!/usr/bin/env python3


# IMPORTS


import sys
import math
import random
# add following directories to class path
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/utilities')

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from utils import Utils
from logger import Logger


# CLASS

'''
    This class handles all things genetic algorithm learning.
'''

class GeneticAlgorithmLearner():


    '''
    CONSTRUCTOR

    args: ?
    '''
    def __init__(self):
        #self.logger = Logger('DEBUG') # configure class-level log level here
        self.training_method_name = 'GA'
        # population size - number of entities evolving over time
        self.population_size = 0
        # maximum number of iterations for entities to evolve
        self.max_iterations = None
        # list of parents in current generation
        self.current_generation = None
        self.current_roullete_wheel = None


    # MAIN METHOD


    #must have already done one iteration of forward prop to have a fitness value
    def create_roullete_wheel(self):
        #test = [4.2, 6.1, 7.1]
        #test = sorted(test)
        test = [x.fitness for x in self.current_generation]
        test = sorted(test)
        numeric_ref = 1
        roullete_wheel = []
        for x in range(len(test)):
            for y in range(numeric_ref):
                roullete_wheel.append(x)
            numeric_ref +=1
        #print('numeric_ref is %s ' % str(numeric_ref))
        #print('numeric_ref is %s ' % str(len(test)))
        #print(roullete_wheel)
        #print(test)
        self.current_roullete_wheel = roullete_wheel
    

    '''
    evolve entities in population over time, until satisfactory solution or max iterations

    INPUT: ?

    OUTPUT:
    	- return best solution found after convergence or loop termination
    '''
    def evolve(self):
    	pass


    # CORE METHODS
    

    '''
    get fitness of particular chromosome - i.e. get accuracy of given network

    INPUT:
    	- nn_data: feedforward network to test given test data
    	- test_data: test data to calculate network accuracy

    OUTPUT:
    	- return fitness score of chromosome (accuracy of network, a float value)
    '''
    def get_fitness(self, nn_data, test_data):
    	pass


    '''
    get mating pool for current iteration

    INPUT: ?

    OUTPUT:
    	- return mating pool for current iteration
    '''
    def get_mating_pool(self):
    	pass


    '''
    match up parents for current iteration

    INPUT: ?

    OUTPUT:
    	- return pairs of parents for current iteration
    '''
    def get_parents(self):
        cast = random.randrange(len(self.current_roullete_wheel))
        parent_one = self.current_generation[self.current_roullete_wheel[cast]]
        cast_two = random.randrange(len(self.current_roullete_wheel))
        #NO ASEXUAL REPRODUCTION
        if self.current_roullete_wheel[cast] == self.current_roullete_wheel[cast_two]:
            cast_two = random.randrange(len(self.current_roullete_wheel))
        parent_two = self.current_generation[self.current_roullete_wheel[cast_two]]

        return parent_one, parent_two


    '''
    crossover, mutation, other random variations used in GA

    INPUT: ?

    OUTPUT:
    	- <void> - do variations on offspring of current parents
    '''
    def do_variations(self, parent_one, parent_two, mutation_rate):
        child = parent_one.copy()
        for parent_one_X, parent_two_Y, child_A in zip(parent_one.weights, parent_two.weights, child.weights):
            for parent_one_M, parent_two_N, child_B in zip(parent_one_X, parent_two_Y, child_A):
                for index in range(parent_one_M.shape[0]):
                    #print("YOUR PArent is %s" % parent_one_M[index])
                    random_val = random.uniform(0,1)
                    if random_val >= .5:
                        child_B[index] = parent_two_N[index]
                    mutation = random.uniform(0,1)
                    if mutation < mutation_rate:
                        child_B[index] = random.uniform(0,1)
        #do the same thing for Biases
        for parent_one_X, parent_two_Y, child_A in zip(parent_one.biases, parent_two.biases, child.biases):
            for parent_one_M, parent_two_N, child_B in zip(parent_one_X, parent_two_Y, child_A):
                for index in range(parent_one_M.shape[0]):
                    random_val = random.uniform(0,1)
                    if random_val >= .5:
                        child_B[index] = parent_two_N[index]
                    mutation = random.uniform(0,1)
                    if mutation < mutation_rate:
                        child_B[index] = random.uniform(-2,2)
        return child


    def get_avg_fitness(self):
        y = 0
        for x in self.current_generation:
            y += x.fitness
        y = str((y/len(self.current_generation)))
        return y


    def do_GA(self, population, mutation_rate, number_of_gens, data):
        if self.current_generation == None:
            self.current_generation = population
        for x in range(number_of_gens):     
        #find fitness
            for fit in self.current_generation:
                fit.find_fitness(data)
                #print("FITNESS OF EACH INDIVIDUAL: %s" % str(fit.fitness))    
            print("Your average fitness is %s " % str(self.get_avg_fitness()))
            child_generation = []
            self.create_roullete_wheel()
            for create_children in range(len(self.current_generation)):
        #create roullet wheel, save it in self.current_roullete wheel
                parent_one, parent_two = self.get_parents()
                #print("PARENT One FITNESS %s " % str(parent_one.fitness))
                #print("PARENT two FITNESS %s " % str(parent_two.fitness))
                child = self.do_variations(parent_one, parent_two, mutation_rate)
                child_generation.append(child)

            self.current_generation = child_generation


    '''
    get offspring from current iteration, setup for next iteration

    INPUT: ?

    OUTPUT:
    	- return offspring from current iteration
    '''
    def get_offspring(self):
    	pass


    # HELPERS


    '''
    convert all matrices in neural network to single vector chromosome
    	- single vector of all weights/biases is GA representation of neural network

    INPUT:
    	- nn_data: object containing list of weights and list of biases for network

    OUTPUT:
    	- return ga_data --> single vector chromosome (GA representation of network)
    '''
    def convert_mat_to_vec(self, nn_data):
    	pass


    '''
    convert single vector chromosome to corresponding list of weight matrices and bias vectors
    - list of weights/biases represents all parameters for entire neural network

    INPUT:
    - ga_data: single vector chromosome holding all weights/biases for network

    OUTPUT:
    - return nn_data --> object containing lists of weights/biases (network representation)
    '''
    def convert_vec_to_mat(self, ga_data):
        pass



# EXECUTE SCRIPT


if __name__ == '__main__':
    '''print('\nrunning GeneticAlgorithmLearner...\n')
    print(random.uniform(0,1))
    our_experiment = NeuralNetwork('car', [6,3,4])
    population = our_experiment.init_population(2, 'car', [6,3,4], 'sigmoid')
    for i in population:
        print("POPUlATIONS MATRIXES %s" % str(i.weights))
        print("biases%s" % str(i.biases))
    ga = GeneticAlgorithmLearner()
    child = ga.do_variations(population[0], population[1], .1)
    print("CHILD %s" % str(child.weights))
    print("HI")
    print(child.biases)'''


    #ga.create_roullete_wheel()



    
