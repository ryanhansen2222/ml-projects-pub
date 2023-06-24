#!/usr/bin/env python3


# IMPORTS


import sys
import random
import numpy as np
# add following directories to class path
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')

# from logger import Logger


# CLASS

'''
    This class handles all things differential evolution.
'''

class DifferentialEvolutionLearner():


    '''
    CONSTRUCTOR

    args:
    '''
    def __init__(self, population, train_data, hyperparams, test_data):
        # self.logger = Logger('DEBUG') # configure class-level log level here

        self.training_method_name = 'DE'
        self.data = train_data
        self.test_data = test_data
        self.population = population
        self.hyperparams = hyperparams
        self.best_individual = None
        self.learning_hyperparams = hyperparams['learning_hyperparams']

        self.run_de(self.hyperparams, self.population)


    '''
    METHOD PURPOSE: This method is the main algorithm that calls all other methods in the class
    It will loop until the exit conditions are reached, either we hit the fitness threshold or max iterations

    INPUT: hyperparams : list of parameters defined in experiment_runner
           population : an array of neural_network instantiantions

    OUTPUT: Will print status updates to console on each successful loop
    TODO: Check that it functions as intended
          Does this need to return something specific to experiment runner??
    '''
    def run_de(self, hyperparams, population):
      #find every individual's base fitness
      self.update_population_fitness()

      pop_avg_acc = 0
      current_iters = 0
      num_of_parents_continuing = 0

      # print("We're about to enter the while loop...")

      #we gotta run this until we hit our accuracy threshold or until we hit the max iterations
      #while(pop_avg_acc < self.learning_hyperparams['de_accuracy_threshold'] or current_iters < self.learning_hyperparams['max_de_iterations']):
      while(current_iters < self.learning_hyperparams['max_de_iterations']):

        #reset parent continuation counter
        num_of_parents_continuing = 0

        #calculate roulette wheel before entering for loop
        roulette_wheel = self.roulette_wheel("wheel")
        max_rand_num = len(roulette_wheel) - 1

        #perform selection with every population member
        #for each member in population call de_selection()
        for individual in range(len(self.population)):
          #choose three other individuals based on fitnesses from roulette wheel
          r1 = individual
          r2 = individual
          r3 = individual
          while(r1 is individual or r2 is individual or r3 is individual):
            r1 = random.randint(0, max_rand_num)
            r2 = random.randint(0, max_rand_num)
            r3 = random.randint(0, max_rand_num)

          # print("Individual is: " + str(individual))
          # print("Random1 is: " + str(r1))
          # print("Random2 is: " + str(r2))
          # print("Random3 is: " + str(r3))
          #do de_selection with the 4 individuals
          child = self.de_selection(individual, roulette_wheel[r1], roulette_wheel[r2], roulette_wheel[r3], self.learning_hyperparams['scaling_factor'])

          #do mutation on the child made from selection
          #child = self.de_mutation(child, self.learning_hyperparams['mutation_rate'])

          #perform the replacement step
          num_of_parents_continuing = self.de_replacement(child, individual, num_of_parents_continuing)

        #after for loop, calculate average fitness of population and increase iteration count
        pop_avg_acc = self.get_avg_pop_fitness()
        current_iters = current_iters + 1

        self.print_status(pop_avg_acc, current_iters, num_of_parents_continuing)

      self.roulette_wheel("best")


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: Roulette wheel will make an array of rankings of indices of best fitting individuals for use in selection.

    INPUT: NONE

    OUTPUT: roulette_wheel : an array of indices that has been sorted and ranked for best fitness
    TODO: Check that it functions as intended
          Drop dummy first entry
          Shortcut for final accuracy calculation
          sorted fitness may be from least to greatest instead of greatest to least right now
    '''
    def roulette_wheel(self, type):

      # print("We're finding the roulette wheel with type: " + type)

      fit_dict = {"index" : 0, "fitness" : 0}
      fitness_array = [fit_dict]

      for individual in range(len(self.population)):
        # print(self.population[individual].fitness)
        fit_dict['index'] = individual
        fit_dict['fitness'] = self.population[individual].fitness
        fitness_array.append(fit_dict.copy())

      fitness_array.pop(0)
      sorted_fitnesses = sorted(fitness_array, key=lambda dct: dct['fitness'])

      # print("This is the sorted array: ")
      # print(sorted_fitnesses)

      if type is "wheel":

        roulette_wheel = []
        rank_measure = 1

        for rank in range(self.learning_hyperparams['population_size']):
          for x in range(rank_measure):
            roulette_wheel.append(sorted_fitnesses[rank]['index'])
            roulette_wheel.append(sorted_fitnesses[rank]['index'])
            roulette_wheel.append(sorted_fitnesses[rank]['index'])
          rank_measure += 1

        # print("This is the roulette wheel: ")
        # print(roulette_wheel)
        # print()
        return roulette_wheel

      elif type is "best":
        self.best_individual = self.population[sorted_fitnesses[len(sorted_fitnesses) - 1]['index']]


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: This method will take in 4 individuals and create the child neural network.
                    It calculates child weights and biases and copies the parent network to place the new matrices
                    in the class. We also use a cross over rate to mix with the single parent from the mutation
                    vector.

    INPUT: i : the individual parent index in the population
           r1, r2, r3 : the randomly selected individuals chosen from the roulette_wheel, they are index values in the population
           sf : the scaling factor used in forming the mutation vector. This is set in the hyperparams

    OUTPUT: child : an array holding two arrays, one of weight and one of biases
    TODO: Check weights and biases matrix structure, I don't know that my calculations are correct.
          Is this mutation vector formation the only mutation I should be doing?
          Check that the method functions as intended.
    '''
    def de_selection(self, i, r1, r2, r3, sf):
      # print("We're performing selection...")

      #given a single member of a population and 3 other random individuals, build the child vector
      child_weights = [0]
      child_biases = [0]

      sf = random.random() * 2

      for w_matrix in range(len(self.population[i].weights)):
        child_weights.append(np.add(self.population[r1].weights[w_matrix], np.multiply(sf, (np.subtract(self.population[r2].weights[w_matrix], self.population[r3].weights[w_matrix])))))

      child_weights.pop(0)
      # print("Found mutation vector weights...")

      for b_matrix in range(len(self.population[i].biases)):
        child_biases.append(np.add(self.population[r1].biases[b_matrix], np.multiply(sf, (np.subtract(self.population[r2].biases[b_matrix], self.population[r3].biases[b_matrix])))))

      child_biases.pop(0)
      # print("Found mutation vector biases...")

      #do cross over between current selected parent and child vector
      for w_matrix in range(len(self.population[i].weights)):
        for row in range(self.population[i].weights[w_matrix].shape[0]):
          for col in range(self.population[i].weights[w_matrix].shape[1]):
            try_crossover = random.random()
            if try_crossover < self.learning_hyperparams['crossover_rate']:
              pass
            else:
              # print("i is: " + str(i))
              # print("w_matrix is: " + str(w_matrix))
              # print("row is: " + str(row))
              # print("col is: " + str(col))
              # print("The value at the w_matrix[row][col] is: " + str(self.population[i].weights[w_matrix][row][col]))
              self.population[i].weights[w_matrix][row][col] = child_weights[w_matrix][row][col]

      # print("Did crossover on weights...")

      for b_matrix in range(len(self.population[i].biases)):
        for row in range(self.population[i].biases[b_matrix].shape[0]):
          for col in range(self.population[i].biases[b_matrix].shape[1]):
            try_crossover = random.random()
            if try_crossover < self.learning_hyperparams['crossover_rate']:
              pass
            else:
              # print("i is: " + str(i))
              # print("w_matrix is: " + str(w_matrix))
              # print("row is: " + str(row))
              # print("col is: " + str(col))
              # print("The value at the w_matrix[row][col] is: " + str(self.population[i].biases[b_matrix][row][col]))
              self.population[i].biases[b_matrix][row][col] = child_biases[b_matrix][row][col]

      # print("Did crossover on biases...")

      child = [child_weights, child_biases]

      return child


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: This method will introduce single point mutations to each weight or bias value in the child arrays

    INPUT: child : an array holding two arrays, one for weights and the other for biases
           mr : the mutation rate, this is a hyperparam

    OUTPUT: child : an array holding two arrays, one for weights and the other for biases
    TODO: Do we even need this mutation step when we are building a mutation vector in selection?
          Check that this method functions as intended
    '''
    # def de_mutation(self, child, mr):
    #   child_weights = child[0]
    #   child_biases = child[1]
    #
    #   for w in range(len(child_weights)):
    #     try_mutate = random.random()
    #     if try_mutate > mr:
    #       pass
    #     else:
    #       child_weights[w] = random.random()
    #
    #   for b in range(len(child_biases)):
    #     try_mutate = random.random()
    #     if try_mutate < mr:
    #       pass
    #     else:
    #       child_biases[b] = random.random()
    #
    #   child = [child_weights, child_biases]
    #
    #   return child

    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: This method will decide which individual stays in the population, the child or the parent.

    INPUT: child : an array holding two arrays, one for weights and the other for biases
           individual : the array index of the parent neural net in the population

    OUTPUT: NONE
    TODO: Check that the method functions as intended
    '''
    def de_replacement(self, child, individual, num_of_parents_continuing):
      # print("We're performing selection...")

      #check fitness in de_replacement to decide if we keep parent or child
      child_nn = self.population[individual].copy()
      child_nn.weights = child[0]
      child_nn.biases = child[1]

      child_nn.find_fitness(self.data)

      if child_nn.fitness < self.population[individual].fitness:
        num_of_parents_continuing += 1
        # print("Parent continued which makes that: " + str(num_of_parents_continuing))
      else:
        self.population[individual] = child_nn
        # print("The child is better...")

      return num_of_parents_continuing


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: Will find the average fitness of the population to determine if we should stop iterating

    INPUT: NONE

    OUTPUT: avg_fit : the average fitness from the entire population.
    TODO: I think this should return an accuracy value rather than average fitness, how do we turn fitness into accuracy?
          Check that this method functions as intended
    '''
    def get_avg_pop_fitness(self):
      fitness_sum = 0
      for individual in self.population:
        fitness_sum += individual.fitness

      avg_fit = fitness_sum / self.learning_hyperparams['population_size']

      return avg_fit


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: This method will initialize the fitness for each individual in the population

    INPUT: NONE

    OUTPUT: NONE
    TODO: Check that this method functions as intended
    '''
    def update_population_fitness(self):
        print('Initializing Population Fitness')
        for individual in self.population:
            individual.find_fitness(self.data)

        print('Population Fitness Updated')


    #-------------------------------------------------------------------------------------------------------------------
    '''
    METHOD PURPOSE: This method will display a result after each iteration of the while loop in run_de

    INPUT: pop_avg : an average fitness of the entire population
           iters : the count of the current number of iterations of the while loop in run_de
           parents : the count of how many parents continued from the previous generation

    OUTPUT: NONE
    TODO: Check that this method functions as intended
    '''
    def print_status(self, pop_avg, iters, parents):
      print("Average Population Fitness: " + str(pop_avg))
      print("Iteration Count: " + str(iters))
      print("Number of Parents Continuing into Next Generation: " + str(parents))
      print("EXIT CONDITION Max Iteration Count: " + str(self.learning_hyperparams['max_de_iterations']))
      #print("Popuation Fitness Threshold: " + str(self.learning_hyperparams['de_accuracy_threshold']))



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning DifferentialEvolutionLearner...\n')
