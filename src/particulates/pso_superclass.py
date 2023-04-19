import pickle

import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from src import analysis, parameter_stuff_and_things as param
from src.particulates import particle as indiv
import src.differentiation.diff_misc as mootils

# import ev_indiv as indiv

multithreading = True
debugmode = False


class POPULATION:
    def __init__(self, max_lim, mu_indivs, sigma, rate_change, sampling_frequency, layers_size,
                 ind_mutations=False):
        self.max_lim = max_lim
        self.mu_indivs = mu_indivs
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
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Positives', 'Negatives'])
        self.test_accuracy_df = DataFrame(
            columns=['Iteration', 'Accuracy', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Positives', 'Negatives'])
        self.fitOverTime = []
        self.ind_mutations = ind_mutations
    #      check if rate_change is a list or a single value
    #     if isinstance(self.rate_change, list):
    #         self.ind_mutations = True
    #     else:
    #         self.ind_mutations = False

    def conception(self):
        self.layers_size.insert(0, self.data_x.shape[1])
        # print("Layers size: " + str(self.layers_size))
        for i in range(self.mu_indivs):
            self.individuals.append(indiv.EVOLUTIONARY_UNIT(self.layers_size, self.data_x, self.data_y))
        self.individuals = [x.initialize_parameters() for x in self.individuals]
        [x.set_weights_and_rates(weights=self.sampling_frequency, rates=self.rate_change) for x in self.individuals]
        # [x.fitness(self.data_x, self.data_y) for x in self.individuals]
            # self.individuals.append(param.init_parameters(self.data_x.shape[0], self.layers_size,
            #                                               sigma=self.sigma[i] if self.ind_mutations else self.sigma))

    def children_production(self, childid, unupdated):
        (self.offspring[childid]).parameters = unupdated.update_params(
            self.sigma if self.ind_mutations else self.sigma[childid % self.mu_indivs])
        if self.offspring[childid].fitness() > unupdated.fitness():
            self.counter += 1


    def findFamily(self, indiv_index):
        choices = np.random.choice(self.mu_indivs, 3, replace=False)
        while choices[0] == indiv_index or choices[1] == indiv_index or choices[2] == indiv_index:
            choices = np.random.choice(self.mu_indivs, 3, replace=False)
        secondary, aunt, uncle = choices[0], choices[1], choices[2]
        return self.individuals[secondary], self.individuals[aunt], self.individuals[uncle]

    def reproduce(self, indiv_index):
        Xp = self.individuals[indiv_index]
        secondary, aunt, uncle = self.findFamily(indiv_index)
        mutated_secondary = secondary.add(uncle.diff(aunt).mul(self.rate_change))
        return Xp.crossover(mutated_secondary, self.sampling_frequency, self.sigma)

    def train_population(self):
        returnModel = None
        self.conception()
        # [x.fitness(self.data_x, self.data_y) for x in ]

        best = self.individuals[0]

        # self.offspring = [indiv.EVOLUTIONARY_UNIT(self.layers_size) for _ in range(self.mu_indivs)]
        prog_bar = tqdm(np.arange(self.max_lim))
        best.fitness(self.data_x, self.data_y)
        iter = 0
        for _ in prog_bar:
            self.individuals.sort(key=lambda x: x.fitness_value, reverse=True)
            best = self.individuals[0]
            self.best_individual = best
            # self.rate_change = 1.08 - best.fitness(self.data_x, self.data_y)
            # self.sampling_frequency = (int) (20/( best.fitness(self.data_x, self.data_y) * 10 + 0.01))
            fit_array = np.array([x.fitness(self.data_x, self.data_y) for x in self.individuals])
            if best.fitness(self.data_x, self.data_y) < 0.9975:
                 [x.update_velocities(self.best_individual) for x in self.individuals]


            analysis.test_accuracy(self, self.data_x, self.data_y, iter,
                                   parameters=best.parameters, layers_size=self.layers_size)
            analysis.test_accuracy(self, self.test_data_x, self.test_data_y, iter,
                                   parameters=best.parameters, layers_size=self.layers_size, test_set=True)

            # prog_bar.set_postfix({'Best': param.fitness(self.data_x, self.data_y, best, self.layers_size), 'tp': self.individuals[0].tp, 'tn': self.individuals[0].tn, 'fp': self.individuals[0].fp, 'fn': self.individuals[0].fn, 'mu': self.mu_indivs, 'fits_range': fit_arr.max() - fit_arr.min()})
            prog_bar.set_postfix(
                {'Best': fit_array.max(), 'Worst': fit_array.min(),
                 'fits_range': fit_array.max() - fit_array.min(),
                 'Curr_Mutation_Rate': self.rate_change})
            iter += 1

        pickle.dump(best, open("./megaRuns/best_model.pkl", "wb"))
        return best



    def predict(self, data_x, target_out_y):
        # predict using the best individual
        [x.fitness(data_x, target_out_y) for x in self.individuals]
        self.best_individual = sorted(self.individuals, key=lambda x: x.fitness_value, reverse=True)[0]
        return self.best_individual.predict_cheaty(data_x, target_out_y)
