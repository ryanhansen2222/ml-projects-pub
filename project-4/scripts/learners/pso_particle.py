#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../../project-3/scripts/logging')

from logger import Logger

import numpy as np
import random


# CLASS

'''
    This class handles all things particle for PSO.
'''

class Particle():


    '''
    CONSTRUCTOR

    - each particle composed of 3 D-dimensional vectors, where D is dimension of the search space:
		- xi --> current position
		- pi --> previous best position
		- vi --> velocity

    ARGS:
    	- start_pos: starting position - randomly initialized
    	- start_vel: starting velocity - randomly initialized
    '''
    def __init__(self, start_pos):
        self.logger = Logger('DEBUG') # configure class-level log level here

        # current position - Neural Net
        self.current_pos = start_pos
        # previous best position - Neural Net
        self.prev_best_pos = self.copy(None)
        # velocity - Different Neural Net
        self.velocity = None
        #Old velocity - used in inertia
        self.oldvelocity= None
        #Fitness - fwdprop of dataset
        self.fitness = None


    '''This method implements the update strategy for the velocity'''
    '''Issue is the velocity method'''

    def update_v(self, globest):
        #Init Velocity
        if(self.velocity == None):
            self.velocity = (np.array(globest.weights)-np.array(self.current_pos.weights),np.array(globest.biases)-np.array(self.current_pos.biases))

        w = .4#Momentum Coefficient
        c1 = 2#Independent Velocity Coefficient
        c2 = 1#Social Velocity Component
         
        
        #Defining weights and bias variables for math
        weights = self.current_pos.weights
        biases = self.current_pos.biases
        lbweights = self.prev_best_pos.weights
        lbbiases = self.prev_best_pos.biases
        gbweights = globest.weights
        gbbiases = globest.biases

        #Update Weights
        for matrix in range(len(self.velocity[0])):
            for val1 in range(len(self.velocity[0][matrix])):
                for val2 in range(len(self.velocity[0][matrix][val1])):
                    #Stochastic Parameters 
                    a = random.random()
                    b = random.random()
                    #Velocity Components
                    winertia = self.velocity[0][matrix][val1][val2]
                    windep = lbweights[matrix][val1][val2]-weights[matrix][val1][val2]
                    wsocial = gbweights[matrix][val1][val2]-weights[matrix][val1][val2]
                    self.velocity[0][matrix][val1][val2]=w*winertia+a*c1*(windep)+b*c2*wsocial            

        #Update Biases
        for vector in range(len(self.velocity[1])):
            for val in range(len(self.velocity[1][vector])):
                #Stochastic Parameters
                a = random.random()
                b = random.random()
                #Velocity Components
                binertia = self.velocity[1][vector][val]
                bindep = lbbiases[vector][val]-biases[vector][val]
                bsocial = gbbiases[vector][val]-weights[vector][val]
                self.velocity[1][vector][val]=w*binertia+a*c1*(bindep)+b*c2*wsocial            


        '''
        #Create velocity components and store as self.velocity
        independent = (np.array(a*c1*np.array(oldweights-weights)),np.array(a*c1*np.array(oldbiases-biases)))

        social = (np.array(c2*b*np.array(globest.weights-weights)),np.array(b*c2*np.array(globest.biases-biases)))

        if(self.oldvelocity != None):
            inertia = (np.array(w*np.array(self.oldvelocity[0])), np.array(w*np.array(self.oldvelocity[1])))
            self.oldvelocity = self.velocity
            self.velocity = (independent[0]+social[0]+inertia[0], independent[1]+social[1]+inertia[1])
        else:
            self.oldvelocity = self.velocity
            
            self.velocity = (independent[0]+social[0], independent[1]+social[1])
        '''


    def update_pos(self):
        '''
        print('Previous Position')
        print(self.current_pos.weights)
        print('Velocity')
        print(self.velocity[0])
        print('New Position')
        '''

        self.current_pos.weights = np.array(self.current_pos.weights) + self.velocity[0] 
        #print(self.current_pos.weights)
        self.current_pos.biases = np.array(self.current_pos.biases) + self.velocity[1]
        

    def update_fitness(self, data):
        x = self.fitness
        self.fitness = self.current_pos.find_fitness(data)
        #print(x, '--->', self.fitness)
        if(self.prev_best_pos.fitness == None):
            self.prev_best_pos.fitness = self.fitness
        elif(self.prev_best_pos.fitness < self.fitness):
            self.prev_best_pos = self.copy(self.fitness)


    def copy(self, fitness):

        copy = self.current_pos.copy()
        copy.fitness = fitness

        return copy
    


# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning PSO Particle...\n')


