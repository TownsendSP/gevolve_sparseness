import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

import analysis
# import ev_indiv as indiv
import parameter_stuff_and_things as param

multithreading = True
debugmode = False


class POPULATION:
    def __init__(self, max_lim, mu_indivs, sigma, lambda_children, rate_change, sampling_frequency, layers_size,
                 ind_mutations=False):
        self.max_lim = max_lim
        self.mu_indivs = mu_indivs
        self.lambda_children = lambda_children
        self.rate_change = rate_change
        self.sampling_frequency = sampling_frequency
        self.layers_size = layers_size
        self.num_layers = len(self.layers_size)
        self.sigma = np.tile(np.array(np.random.normal(loc=1, scale=sigma, size=self.num_layers + 1)).clip(0, 100),
                             (self.mu_indivs, 1)) if ind_mutations else sigma
        self.individuals = []
        self.data_x = None
        self.data_y = None
        self.test_data_x = None
        self.test_data_y = None
        self.best_individual = None
        self.offspring = []
        self.counter = 0
        self.train_accuracy_df = DataFrame(
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.test_accuracy_df = DataFrame(
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives'])
        self.fitOverTime = []
        self.ind_mutations = ind_mutations

    def conception(self):
        self.layers_size.insert(0, self.data_x.shape[1])
        # print("Layers size: " + str(self.layers_size))
        for i in range(self.mu_indivs):
            # self.individuals.append(indiv.EVOLUTIONARY_UNIT(self.layers_size))
            self.individuals.append(param.init_parameters(self.data_x.shape[0], self.layers_size,
                                                          sigma=self.sigma[i] if self.ind_mutations else self.sigma))

        # for i in range(len(self.individuals)):
        #     # self.individuals[i].layers_size.insert(0, self.data_x.shape[1])
        #     self.individuals[i].data_x = self.data_x
        #     self.individuals[i].data_y = self.data_y
        #     self.individuals[i].init_params_v2()

    def children_production(self, childid, unupdated):
        (self.offspring[childid]).parameters = unupdated.update_params(
            self.sigma if self.ind_mutations else self.sigma[childid % self.mu_indivs])
        if self.offspring[childid].fitness() > unupdated.fitness():
            self.counter += 1

    def reproduce(self):
        self.counter = 0
        self.offspring = np.repeat(self.individuals, self.lambda_children // self.mu_indivs)
        print(str.format("lambda_children: {0}, mu_indivs: {1}, offspring: {2}, offspring 0 fitness: {3}",
                         self.lambda_children, self.mu_indivs, len(self.offspring),
                         self.offspring[0].fitness())) if debugmode else None
        # spoons = [param.update_parameters(x, self.sigma, self.layers_size) for x in self.offspring]
        spoons = [param.update_parameters(self.offspring[i], self.sigma[i % self.mu_indivs], self.layers_size) for i in
                  range(self.lambda_children)]
        self.offspring = spoons

        fitArr = []
        for spoon in spoons:
            fitArr.append(param.fitness(self.data_x, self.data_y, spoon, self.layers_size))
            # fitArr.append(self.offspring[i].fitness())
        self.fitOverTime.append(fitArr)

        print(str.format("offspring 0 fitness: {0}", self.offspring[0].fitness())) if debugmode else None
        max_parent_fitness = max(
            [param.fitness(self.data_x, self.data_y, x, self.layers_size) for x in self.individuals])
        forks = len([x for x in self.offspring if
                     param.fitness(self.data_x, self.data_y, x, self.layers_size) > max_parent_fitness])
        return forks

    def train_population(self):
        self.conception()
        prog_bar = tqdm(np.arange(self.max_lim))
        iter = 0
        for _ in prog_bar:
            self.counter = 0
            self.offspring = [param.init_parameters(self.data_x.shape[0], self.layers_size, self.sigma[0]) for _ in
                              range(self.lambda_children)]
            print(str(type(self.offspring))) if debugmode else None
            best = self.individuals[0]
            original_fitnesses = [param.fitness(self.data_x, self.data_y, x, self.layers_size) for x in
                                  self.individuals]
            for n in range(self.sampling_frequency):
                analysis.test_accuracy(self, self.data_x, self.data_y, iter,
                                       parameters=best, layers_size=self.layers_size)
                analysis.test_accuracy(self, self.test_data_x, self.test_data_y, iter,
                                       parameters=best, layers_size=self.layers_size, test_set=True)
                improved = self.reproduce()
                self.counter += improved
                family = np.concatenate((self.individuals, self.offspring))
                # family = self.individuals.append + self.offspring
                print(str.format("Family {0}\n", str(family))) if debugmode else None
                self.individuals = sorted(family,
                                          key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size),
                                          reverse=True)
                # self.individuals = sorted(family, key=lambda x: x.fitness())
                # best = max(self.individuals, key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size))
                best = self.individuals[0]

                fit_arr = np.array(
                    [param.fitness(self.data_x, self.data_y, x, self.layers_size) for x in self.individuals])
                self.individuals = self.individuals[0:self.mu_indivs]
                # prog_bar.set_postfix({'Best': param.fitness(self.data_x, self.data_y, best, self.layers_size), 'tp': self.individuals[0].tp, 'tn': self.individuals[0].tn, 'fp': self.individuals[0].fp, 'fn': self.individuals[0].fn, 'mu': self.mu_indivs, 'fits_range': fit_arr.max() - fit_arr.min()})
                prog_bar.set_postfix(
                    {'Best': param.fitness(self.data_x, self.data_y, best, self.layers_size), 'mu': self.mu_indivs,
                     'fits_range': fit_arr.max() - fit_arr.min()})
                iter += 1

            if self.ind_mutations:
                for i in range(len(self.individuals)):
                    if param.fitness(self.data_x, self.data_y, self.individuals[i], self.layers_size) > \
                            original_fitnesses[i]:
                        for j in range(len(self.sigma[i])):
                            self.sigma[i][j] = (1 - self.rate_change[j])
                    else:
                        for j in range(len(self.sigma[i])):
                            self.sigma[i][j] = (1 + self.rate_change[j])

            else:

                if self.counter < (self.sampling_frequency * (self.lambda_children / 5)):
                    self.sigma = (1 - self.rate_change)
                elif self.counter > (self.sampling_frequency * (self.lambda_children / 5)):
                    self.sigma = (1 + self.rate_change)

            # self.sigma += 1 if self.counter > (self.sampling_frequency * (self.lambda_children / 5)) else -1
            # sorted_params = sorted(self.individuals, key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size), reverse=True)[0]
        return self.individuals[0]

    def train_population_indiv_mutation(self):
        self.conception()
        prog_bar = tqdm(np.arange(self.max_lim))
        iter = 0
        for _ in prog_bar:
            self.counter = 0
            self.offspring = [param.init_parameters(self.data_x.shape[0], self.layers_size, self.sigma) for _ in
                              range(self.lambda_children)]
            print(str(type(self.offspring))) if debugmode else None
            best = self.individuals[0]
            for n in range(self.sampling_frequency):
                analysis.test_accuracy(self, self.data_x, self.data_y, iter,
                                       parameters=best, layers_size=self.layers_size)
                analysis.test_accuracy(self, self.test_data_x, self.test_data_y, iter,
                                       parameters=best, layers_size=self.layers_size, test_set=True)
                improved = self.reproduce()
                self.counter += improved
                family = np.concatenate((self.individuals, self.offspring))
                # family = self.individuals.append + self.offspring
                print(str.format("Family {0}\n", str(family))) if debugmode else None
                self.individuals = sorted(family,
                                          key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size),
                                          reverse=True)
                # self.individuals = sorted(family, key=lambda x: x.fitness())
                # best = max(self.individuals, key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size))
                best = self.individuals[0]

                # if self.best_individual is not None:
                #     if self.best_individual.fitness() > self.individuals[0].fitness():
                #         self.best_individual = self.individuals[0].fitness()
                # elif self.best_individual is None:
                #     self.best_individual = self.individuals[0]
                # self.best_individual = self.individuals[0]
                fit_arr = np.array(
                    [param.fitness(self.data_x, self.data_y, x, self.layers_size) for x in self.individuals])
                self.individuals = self.individuals[0:self.mu_indivs]
                # prog_bar.set_postfix({'Best': param.fitness(self.data_x, self.data_y, best, self.layers_size), 'tp': self.individuals[0].tp, 'tn': self.individuals[0].tn, 'fp': self.individuals[0].fp, 'fn': self.individuals[0].fn, 'mu': self.mu_indivs, 'fits_range': fit_arr.max() - fit_arr.min()})
                prog_bar.set_postfix(
                    {'Best': param.fitness(self.data_x, self.data_y, best, self.layers_size), 'mu': self.mu_indivs,
                     'fits_range': fit_arr.max() - fit_arr.min()})
                # if self.individuals[0].fitness() >= 100:
                #     break
                # InfoString = str.format("Gen {0} - Best fit:{1} - Worst fit:{2} - Num_indivs:{3} - Num_offspring:{4}",
                #                         k, self.individuals[0].fitness(), self.individuals[-1].fitness(),
                #                         len(self.individuals), len(self.offspring))
                iter += 1

            if self.counter < (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 - self.rate_change)
            elif self.counter > (self.sampling_frequency * (self.lambda_children / 5)):
                self.sigma = (1 + self.rate_change)
            # sorted_params = sorted(self.individuals, key=lambda x: param.fitness(self.data_x, self.data_y, x, self.layers_size), reverse=True)[0]

    def predict(self, data_x, target_out_y):
        # predict using the best individual
        self.best_individual = sorted(self.individuals, key=lambda x: x.fitness(), reverse=True)[0]
        return self.best_individual.predict_cheaty(data_x, target_out_y)
