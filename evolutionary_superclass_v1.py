import threading

import numpy as np
import pandas as pd
from numpy import random
from tqdm.auto import tqdm

import analysis
import ev_indiv as indiv

multithreading = True
debugmode = False


class POPULATION:
    def __init__(self, max_lim, mu_indivs, sigma, lambda_children, rate_change, sampling_frequency, layers_size):
        self.max_lim = max_lim
        self.mu_indivs = mu_indivs
        self.sigma = sigma
        self.lambda_children = lambda_children
        self.rate_change = rate_change
        self.sampling_frequency = sampling_frequency
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
        self.train_accuracy_df = pd.DataFrame(columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.test_accuracy_df = pd.DataFrame(columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.fitOverTime = []

    def initialize_population(self):
        self.layers_size.insert(0, self.data_x.shape[1])
        for i in range(self.mu_indivs):
            self.individuals.append(indiv.EVOLUTIONARY_UNIT(self.layers_size))

        for i in range(len(self.individuals)):
            # self.individuals[i].layers_size.insert(0, self.data_x.shape[1])
            self.individuals[i].data_x = self.data_x
            self.individuals[i].data_y = self.data_y
            self.individuals[i].init_params_v2()

    def birth(self, parameters):
        newborn = indiv.EVOLUTIONARY_UNIT(self.layers_size)
        newborn.parameters = parameters
        newborn.data_x = self.data_x
        newborn.data_y = self.data_y
        # print("newborn parameters: " + str(newborn.parameters)) if debugmode else None
        return newborn

    def children_production(self, childid, unupdated):
        (self.offspring[childid]).parameters = unupdated.update_params(self.sigma)
        if self.offspring[childid].fitness() > unupdated.fitness():
            self.counter += 1

    def pop_reproduce(self):
        self.counter = 0
        self.offspring = np.repeat(self.individuals, self.lambda_children//self.mu_indivs)
        print(str.format("lambda_children: {0}, mu_indivs: {1}, offspring: {2}, offspring 0 fitness: {3}", self.lambda_children, self.mu_indivs, len(self.offspring), self.offspring[0].fitness())) if debugmode else None
        [x.update_params(self.sigma) for x in self.offspring]

        fitArr = []
        for i in range(len(self.offspring)):
            fitArr.append(self.offspring[i].fitness())
        self.fitOverTime.append(fitArr)



        print(str.format("offspring 0 fitness: {0}", self.offspring[0].fitness())) if debugmode else None
        max_parent_fitness = max([x.fitness() for x in self.individuals])
        forks = len([x for x in self.offspring if x.fitness() > max_parent_fitness])
        return forks

    def train_population(self):
        self.initialize_population()
        prog_bar = tqdm(np.arange(self.max_lim))
        for k in prog_bar:
            self.counter = 0
            self.offspring = [indiv.EVOLUTIONARY_UNIT(self.layers_size, data_x=self.data_x, data_y=self.data_y) for _ in
                              range(self.lambda_children)]
            print(str(type(self.offspring))) if debugmode else None
            for n in range(self.sampling_frequency):
                improved = self.pop_reproduce()

                print(str(self.offspring)) if debugmode else None
                print(str(self.individuals)) if debugmode else None
                family = np.concatenate((self.individuals, self.offspring))
                # family = self.individuals.append + self.offspring
                print(str.format("Family {0}\n", str(family))) if debugmode else None
                self.individuals = sorted(family, key=lambda x: x.fitness(), reverse=True)
                # self.individuals = sorted(family, key=lambda x: x.fitness())
                best = max(self.individuals, key=lambda x: x.fitness())
                first = self.individuals[0]

                # if self.best_individual is not None:
                #     if self.best_individual.fitness() > self.individuals[0].fitness():
                #         self.best_individual = self.individuals[0].fitness()
                # elif self.best_individual is None:
                #     self.best_individual = self.individuals[0]
                # self.best_individual = self.individuals[0]
                fit_arr = np.array([x.fitness() for x in self.individuals])
                self.individuals = self.individuals[0:self.mu_indivs]
                prog_bar.set_postfix(dict(Best=best.fitness(), tp=self.individuals[0].tp, tn=self.individuals[0].tn,
                                          fp=self.individuals[0].fp, fn=self.individuals[0].fn, mu=self.mu_indivs,
                                          fits_range=fit_arr.max() - fit_arr.min()))
                # if self.individuals[0].fitness() >= 100:
                #     break
                # InfoString = str.format("Gen {0} - Best fit:{1} - Worst fit:{2} - Num_indivs:{3} - Num_offspring:{4}",
                #                         k, self.individuals[0].fitness(), self.individuals[-1].fitness(),
                #                         len(self.individuals), len(self.offspring))

            if self.counter < (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 - self.rate_change)
            elif self.counter > (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 + self.rate_change)
            analysis.test_accuracy(self, self.data_x, self.data_y, k)
            analysis.test_accuracy(self, self.test_data_x, self.test_data_y, k, test_set=True)

    def train_population(self):
        self.initialize_population()
        prog_bar = tqdm(np.arange(self.max_lim))
        for k in prog_bar:
            self.counter = 0
            self.offspring = [indiv.EVOLUTIONARY_UNIT(self.layers_size, data_x=self.data_x, data_y=self.data_y) for _ in
                              range(self.lambda_children)]
            print(str(type(self.offspring))) if debugmode else None
            for n in range(self.sampling_frequency):
                self.pop_reproduce()

                print(str(self.offspring)) if debugmode else None
                print(str(self.individuals)) if debugmode else None
                family = np.concatenate((self.individuals, self.offspring))
                # family = self.individuals.append + self.offspring
                print(str.format("Family {0}\n", str(family))) if debugmode else None
                self.individuals = sorted(family, key=lambda x: x.fitness(), reverse=True)
                # self.individuals = sorted(family, key=lambda x: x.fitness())
                best = max(self.individuals, key=lambda x: x.fitness())
                first = self.individuals[0]

                # if self.best_individual is not None:
                #     if self.best_individual.fitness() > self.individuals[0].fitness():
                #         self.best_individual = self.individuals[0].fitness()
                # elif self.best_individual is None:
                #     self.best_individual = self.individuals[0]
                # self.best_individual = self.individuals[0]
                fit_arr = np.array([x.fitness() for x in self.individuals])
                self.individuals = self.individuals[0:self.mu_indivs]
                prog_bar.set_postfix({'Best': best.fitness(), 'tp': self.individuals[0].tp, 'tn': self.individuals[0].tn, 'fp': self.individuals[0].fp, 'fn': self.individuals[0].fn, 'mu': self.mu_indivs, 'fits_range': fit_arr.max() - fit_arr.min()})
                # if self.individuals[0].fitness() >= 100:
                #     break
                # InfoString = str.format("Gen {0} - Best fit:{1} - Worst fit:{2} - Num_indivs:{3} - Num_offspring:{4}",
                #                         k, self.individuals[0].fitness(), self.individuals[-1].fitness(),
                #                         len(self.individuals), len(self.offspring))

            if self.counter < (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 - self.rate_change)
            elif self.counter > (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 + self.rate_change)
            analysis.test_accuracy(self, self.data_x, self.data_y, k)
            analysis.test_accuracy(self, self.test_data_x, self.test_data_y, k, test_set=True)

    def train_population_v1(self):
        self.initialize_population()
        prog_bar = tqdm(np.arange(self.max_lim))
        for k in prog_bar:
            self.counter = 0
            self.offspring = [indiv.EVOLUTIONARY_UNIT(self.layers_size, data_x=self.data_x, data_y=self.data_y) for _ in
                              range(self.lambda_children)]
            print(str(type(self.offspring))) if debugmode else None
            for n in range(self.sampling_frequency):
                offspring = np.random.choice(self.individuals, self.lambda_children)
                if multithreading:
                    threads = []
                    for i in range(self.lambda_children):
                        t = threading.Thread(target=self.children_production, args=(i, self.best_individual if self.best_individual is not None else offspring[i]))
                        # t = threading.Thread(target=self.children_production, args=(i, offspring[i]))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                # for k in tqdm(range(self.max_lim)):
                #     self.counter = 0
                #     self.offspring = [indiv.EVOLUTIONARY_UNIT(self.layers_size, data_x=self.data_x, data_y=self.data_y) for _ in
                #                       range(self.lambda_children)]
                #     print(str(type(self.offspring))) if debugmode else None
                #     for n in range(self.sampling_frequency):
                #         offspring = np.random.choice(self.individuals, self.lambda_children)
                #         if multithreading:
                #             threads = []
                #             for i in range(self.lambda_children):
                #                 t = threading.Thread(target=self.children_production, args=(i, offspring[i]))
                #                 threads.append(t)
                #                 t.start()
                #             for t in threads:
                #                 t.join()  # self.counter = len(updated_offspring)
                #
                #         else:
                #             updated_offspring = []
                #             for unupdated in offspring:
                #                 new_child = self.birth(unupdated.update_params(self.sigma))
                #                 if new_child.fitness() > unupdated.fitness():
                #                     self.counter += 1
                #                 updated_offspring.append(new_child)
                print(str(self.offspring)) if debugmode else None
                print(str(self.individuals)) if debugmode else None
                family = np.concatenate((self.individuals, self.offspring))
                # family = self.individuals.append + self.offspring
                print(str.format("Family {0}\n", str(family))) if debugmode else None
                self.individuals = sorted(family, key=lambda x: x.fitness(), reverse=True)
                # self.individuals = sorted(family, key=lambda x: x.fitness())
                best = max(self.individuals, key=lambda x: x.fitness())

                # if self.best_individual is not None:
                #     if self.best_individual.fitness() > self.individuals[0].fitness():
                #         self.best_individual = self.individuals[0].fitness()
                # elif self.best_individual is None:
                #     self.best_individual = self.individuals[0]
                # self.best_individual = self.individuals[0]
                fit_arr = np.array([x.fitness() for x in self.individuals])
                self.individuals = self.individuals[0:self.mu_indivs]
                prog_bar.set_postfix({'Best': best.fitness(), 'tp': self.individuals[0].tp, 'tn': self.individuals[0].tn, 'fp': self.individuals[0].fp, 'fn': self.individuals[0].fn, 'mu': self.mu_indivs, 'fits': fit_arr})
                # if self.individuals[0].fitness() >= 100:
                #     break
                # InfoString = str.format("Gen {0} - Best fit:{1} - Worst fit:{2} - Num_indivs:{3} - Num_offspring:{4}",
                #                         k, self.individuals[0].fitness(), self.individuals[-1].fitness(),
                #                         len(self.individuals), len(self.offspring))

            if self.counter < (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma *= (1 - self.rate_change)
            elif self.counter > (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma *= (1 + self.rate_change)
            analysis.test_accuracy(self, self.data_x, self.data_y, k)
            analysis.test_accuracy(self, self.test_data_x, self.test_data_y, k, test_set=True)

    def predict(self, data_x, target_out_y):
        # predict using the best individual
        self.best_individual = sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)[0]
        return self.best_individual.predict_cheaty(data_x, target_out_y)
