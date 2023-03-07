import numpy as np
import pandas as pd
from src import parameter_stuff_and_things as param
from src import processing as pro
import pickle
import src.analysis as analysis

def reproduce_two_point_crossover(parent1, parent2):
    p = len(parent2.genes)
    b, a = np.array(parent2.genes).flatten().tolist(), np.array(parent1.genes).flatten().tolist()
    x = np.random.randint(0, len(a) - 1)
    # set y to a random number at least one away from x
    y = np.random.choice(np.delete([y for y in range(len(a))], [x + x - 1 for x in range(3)]))
    x, y = (x, y) if x < y else (y, x)
    y = a[x:y]
    a[x:y], b[x:y] = b[x:y], y
    a, b = np.array_split(a, p), np.array_split(b, p)
    return genetic_child(a), genetic_child(b)

def produce_new_genes(x, y):
    a = np.zeros((x, y))
    a.flat[np.random.choice(x * y, int(x * y * 0.1), replace=False)] = 1
    return a.tolist()

class genetic_child():
    def __init__(self, genes = None, params = None, mutate = True):
        self.num_layers = [16, 34, 1]
        self.genes = produce_new_genes(self.num_layers[1], self.num_layers[0]) if genes is None else genes
        self.fitness = None
        self.params = params
        self.masked_params = None
        self.accuracy = None

    def mutate(self, params = None):
        if params is not None and self.params is None:
            self.params = params
        a = [np.random.randint(0, len(self.genes)), np.random.randint(0, len(self.genes[0]))]
        self.genes[a[0]][a[1]] = abs(self.genes[a[0]][a[1]] - 1)

    def gen_fitness(self, data_x, data_y, params = None):
        if self.params is None and params is None:
            raise Exception("No parameters passed to fitness function")
        if self.params is None:
            self.params = params
        # else:
        #     self.params["W1"] = params["W1"] * self.genes
        self.masked_params = self.params.copy()
        self.masked_params["W1"] = self.params["W1"] * self.genes

        self.accuracy = param.accuracy(data_x, data_y, self.masked_params, self.num_layers)
        self.fitness = self.accuracy

        # # self.params["W1"] = params["W1"] * self.genes
        # self.accuracy = param.accuracy(data_x, data_y, self.params, self.num_layers)

    def crossover(self, parent2):
        p = len(parent2.genes)
        b, a = np.array(parent2.genes).flatten().tolist(), np.array(self.genes).flatten().tolist()
        x = np.random.randint(0, len(a) - 1)
        # set y to a random number at least one away from x
        z = np.random.choice(np.delete([y for y in range(len(a))], [x + x - 1 for x in range(3)]))
        # print(str.format("point 1: {0}, point 2: {1}", x, z))
        w = [x, z] if x < z else [z, x]
        q = a[w[0]:w[1]]
        a[w[0]:w[1]], b[w[0]:w[1]] = b[w[0]:w[1]], q

        a, b = np.array_split(a, p), np.array_split(b, p)
        return [genetic_child(a), genetic_child(b)]

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self, f)



