#!/usr/bin/env python3


# IMPORTS

import numpy as np
import sys
# add following directories to class path
sys.path.append('../../../project-3/scripts/networks')
sys.path.append('../../../project-3/scripts/logging')


from logger import Logger
from pso_particle import Particle

# CLASS

'''
    This class handles all things particle swarm optimization.
'''
class ParticleSwarmOptimizationLearner():


    '''
    CONSTRUCTOR

    args:
    '''
    def __init__(self, pop, train_data):
        self.logger = Logger('DEBUG') # configure class-level log level here

        self.training_method_name = 'PSO'
        #Training data
        self.data = train_data
        #A list of particles, each representing a neural net + extra info
        self.swarm = self.swarmify(pop)
        # A Neural Net instance with weights that yield best historical results
        self.bestloc = None
        
        
    '''
    Convert data population to Particle/Swarm Data structure
    '''
    def swarmify(self, population):
        print('Converting to particles')
        swarm = []
        for x in population:
            s = Particle(x)
            swarm.append(s)
        print('Converted to particles')
        return swarm


    def run_swarm_learner(self):

        iterations = 0
        maxiter = 30
        #initialize swarm fitness
        self.update_swarm_fitness()
        
        self.bestloc = self.swarm[0].copy(self.swarm[0].fitness)
        while(iterations <maxiter):
            print(int(100*iterations/maxiter),'% done with current CV Fold')
            #self.update_swarm_fitness()
            self.update_swarm_vel()
            self.update_swarm_pos()
            iterations = iterations + 1
        print('Best Neural Network fitness')
        print(self.bestloc.fitness)
        return self.bestloc
        #print(self.bestloc.weights)
               
         
    def update_swarm_fitness(self):
        print('Initializing Swarm Fitness')
        for particle in self.swarm:
            particle.update_fitness(self.data)
            
        print('Swarm Fitness Updated')


    def update_swarm_pos(self):

        for particle in self.swarm:
            
            particle.update_pos()
            particle.update_fitness(self.data)
            #Update Best known location global
         
            #print('Particle Fitness')
            #print(particle.fitness)
            if(particle.fitness > self.bestloc.fitness):
                print('Found better global fitness: ', self.bestloc.fitness, ' ---> ', particle.fitness)
                self.bestloc = particle.copy(particle.fitness)


    def update_swarm_vel(self):
        for particle in self.swarm:
            particle.update_v(self.bestloc)

             

''' What do i still need ---
Bounding box, velocity generation strategy,''' 



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning ParticleSwarmOptimizationLearner...\n')

    
