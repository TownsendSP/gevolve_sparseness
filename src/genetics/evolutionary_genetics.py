import itertools

import numpy
import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from src import analysis, parameter_stuff_and_things as param

# import ev_indiv as indiv
import src.genetics.genetic_individual as indiv
import numpy.random as npr
multithreading = True
debugmode = False


class POPULATION:
    def __init__(self, max_lim, mu_indivs, layers_size, source_params):
        self.source_params = source_params
        self.max_lim = max_lim
        self.mu_indivs = mu_indivs
        self.layers_size = layers_size
        self.num_layers = len(self.layers_size)
        self.individuals = []
        self.data_x = None
        self.data_y = None
        self.test_data_x = None
        self.test_data_y = None
        self.best_individual = None
        self.offspring = []
        self.counter = 0
        self.best_history = []
        self.train_accuracy_df = DataFrame(
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.test_accuracy_df = DataFrame(
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.fitOverTime = []

    def conception(self):
        self.individuals = [indiv.genetic_child(params=self.source_params) for x in range(self.mu_indivs)]
        [x.gen_fitness(self.data_x, self.data_y) for x in self.individuals]

    def roulette(self):
        total_fitness = sum([c.fitness for c in self.individuals])
        selection_probs = [c.fitness / total_fitness for c in self.individuals]
        return npr.choice(len(self.individuals), p=selection_probs)

    def roulette_twice(self):
        first, second = self.roulette(), self.roulette()
        while first == second:
            second = self.roulette()
        return self.individuals[first], self.individuals[second]

    def sexual_reproduction(self):
        parent1, parent2 = self.roulette_twice()
        return parent1.crossover(parent2)

    def reproduce(self):
        # self.individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        gremlins = numpy.array([self.sexual_reproduction() for x in range(self.mu_indivs - self.mu_indivs // 10)]).flatten().tolist() + [self.individuals[self.roulette()] for x in range(self.mu_indivs // 10)]
        [ankle_biter.mutate(params=self.source_params) for ankle_biter in gremlins]
        # offspring = [ankle_biter.mutate(params=self.source_params) for ankle_biter in
        #              numpy.array([
        #                  [self.sexual_reproduction() for x in range(self.mu_indivs - self.mu_indivs // 10)],
        #                  [self.individuals[self.roulette()] for x in range(self.mu_indivs // 10)]]).flatten().tolist()]
        [x.gen_fitness(self.data_x, self.data_y) for x in gremlins]
        self.individuals = gremlins

    def train_population(self):
        self.conception()
        prog_bar = tqdm(np.arange(self.max_lim))
        self.best_individual = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[0]
        # print(str.format("Initial Best Fitness: {0}", self.best_individual.fitness))
        iter = 0
        for _ in prog_bar:
            if self.best_individual.fitness > 0.998:
                self.best_history.append(self.best_individual.genes)
                print(str.format("Best Fitness: {0}", self.best_individual.fitness))

            else:


                self.reproduce()
                if self.best_individual.fitness < sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[0].fitness:
                    self.best_individual = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[0]
                else:
                    self.individuals.append(self.best_individual)

                self.best_history.append(self.best_individual.genes)
                # analysis.test_accuracy(self, self.data_x, self.data_y, iter,
                #                        parameters=self.best_individual.masked_params, layers_size=self.layers_size)
                # analysis.test_accuracy(self, self.test_data_x, self.test_data_y, iter,
                #                        parameters=self.best_individual.masked_params, layers_size=self.layers_size, test_set=True)
                prog_bar.set_postfix( {'Best': self.best_individual.fitness, 'mu': self.mu_indivs})

            iter += 1

        return self.best_individual.masked_params

    def predict(self, data_x, target_out_y):
        # predict using the best individual
        # best_individual = sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)[0]
        return self.best_individual.predict_cheaty(data_x, target_out_y)
