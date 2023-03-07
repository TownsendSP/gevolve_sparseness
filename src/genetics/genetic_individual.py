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
    return genetic_child(parent1, parent2, a), genetic_child(parent1, parent2, b)


class genetic_child():
    def __init__(self, parent1, parent2, genes = None):
        self.parent1 = parent1
        self.parent2 = parent2
        self.genes = genes
        self.fitness = None
        self.num_layers = [16, 34, 1]
        self.params = None

    def two_point_crossover_per_gene(self, parent1, parent2):
        for i in range (len(parent1)):
            arr1 = parent1[i]
            arr2 = parent2[i]
            point1 = np.random.randint(0, len(arr1) - 1)
            point2 = np.random.randint(point1, len(arr2))
            tempFrom1 = arr1[point1:point2]
            arr1[point1:point2] = arr2[point1:point2]
            arr2[point1:point2] = tempFrom1

    def two_point_crossover_whole_gene(self, parent1, parent2):
        point1 = np.random.randint(0, len(parent1) - 1)
        point2 = np.random.randint(point1, len(parent2))
        tempFrom1 = parent1[point1:point2]
        parent1[point1:point2] = parent2[point1:point2]
        parent2[point1:point2] = tempFrom1

    def two_point_crossover(self):
        p = len(self.parent2.genes)
        b, a = np.array(self.parent2.genes).flatten().tolist(), np.array(self.parent1.genes).flatten().tolist()
        x = np.random.randint(0, len(a) - 1)
        # set y to a random number at least one away from x
        y = np.random.choice(np.delete([y for y in range(len(a))], [x + x - 1 for x in range(3)]))
        x, y = (x, y) if x < y else (y, x)
        y = a[x:y]
        a[x:y], b[x:y] = b[x:y], y
        a, b = np.array_split(a, p), np.array_split(b, p)

    def mutate(self):
        a = [np.random.randint(0, len(self.genes)), np.random.randint(0, len(self.genes[0]))]
        self.genes[a[0]][a[1]] = abs(self.genes[a[0]][a[1]] - 1)


    def fitness(self, data_x, data_y, params):
        # self.params["W1"]
        # multiply each weight by the corresponding gene
        self.params["W1"] = params["W1"] * self.genes
        accuracy = param.accuracy(data_x, data_y, self.params, self.num_layers)


