import numpy as np
import pandas as pd

global sig

class EVOLUTIONARY_UNIT:
    def __init__(self, layers_size, data_x=None, data_y=None):
        self.parameters = {}
        self.velocities = {}
        self.layers_size = layers_size
        self.num_layers = len(layers_size)-1
        self.data_x = data_x
        self.data_y = data_y
        self.fitness_value = None
        self.sigma = 0.6
        self.mutation_weights = [0.3, 0.3, 0.4]
        self.mutation_rate = 0.1
        self.predictions_cache = [None, None]
        self.n = self.data_x.shape[0] if self.data_x is not None else None

    def set_weights_and_rates(self, weights, rates):
        self.mutation_weights = weights
        self.mutation_rate = rates

    # <editor-fold desc="MathStuff">
    def diff(self, other):
    #     calculates vector difference between self and other
        difference = EVOLUTIONARY_UNIT(self.layers_size)
        difference.parameters = self.parameters.copy()
        for layer in range(1, len(self.layers_size)):
            difference.parameters["W" + str(layer)] -= other.parameters["W" + str(layer)]
        return difference

    def __mul__(self, factor):
    #     multiplies every parameter in self by factor, returns new object
        product = EVOLUTIONARY_UNIT(self.layers_size)
        product.parameters = self.parameters.copy()
        for layer in range(1, len(self.layers_size)):
            product.parameters["W" + str(layer)] *= factor
        return product

    def __add__(self, other):
        # adds every parameter in self by factor, returns new object
        sum = EVOLUTIONARY_UNIT(self.layers_size)
        sum.parameters = self.parameters.copy()
        sporkThing = {}
        if isinstance(other, EVOLUTIONARY_UNIT):
            sporkThing = other.parameters
        else:
            sporkThing = other

        for layer in range(1, len(self.layers_size)):
            sum.parameters["W" + str(layer)] += sporkThing["W" + str(layer)]
        return sum
    # </editor-fold>

    def __str__(self):
        return str.format("layers_size: {0}\nnum_layers: {1}\nparameters shape: {2}\nfitness: {3}",
                          self.layers_size, self.num_layers, self.parameters["W1"].shape, self.fitness_value)

    def select_data(self, n):
        # select n random values from np arrs self.data_x and the corresponding values from self.data_y
        #create a new data_x and data_y with the selected values
        indices = np.random.choice(self.data_x.shape[0], n, replace=False)
        data_x = np.array([self.data_x[i] for i in indices])
        data_y = np.array([self.data_y[i] for i in indices])
        return data_x, data_y

    def init_params_zeroes(self):
        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = np.zeros((self.layers_size[layer], self.layers_size[layer - 1]))
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))

    def init_mut_mask(self, mutation_rate):
        # randomly mutates n% of the weights and biases
        for layer in range(1, len(self.layers_size)):
            mut_mat = np.random.normal(loc=0, scale=self.sigma, size=(self.layers_size[layer],self.layers_size[layer - 1]))
            mask_mat = np.random.rand(*mut_mat.shape)<(1-mutation_rate)
            mut_mat[mask_mat] = 0
            self.parameters["W" + str(layer)] = mut_mat

    def mutate(self, mutation_rate):
        # randomly mutates n% of the weights and biases
        mutator = EVOLUTIONARY_UNIT(self.layers_size)
        mutator.init_mut_mask(mutation_rate)
        self += mutator

    def update_velocities(self, global_best):
        mutation_rate = self.mutation_rate
    #     performs three-parent weighted averaging recombination combined with random mutation to update velocities
        w = self.mutation_weights
        # print("w: ", w)
        # print("self: ", self)
        # print("global_best: ", global_best)
        # print("self.best_params: ", self.best_params)
        new_params = self * w[0] + global_best * w[1] + self.best_params * w[2]
        new_params.mutate(mutation_rate)

        # select 100 random values from self.data_x and the corresponding values from self.data_y
        testx, testy = self.select_data(100)
        if new_params.fitness(testx, testy, override=True) > self.fitness_value:
            new_fitness = new_params.fitness(self.data_x, self.data_y, override=True)
            if new_fitness > self.fitness(self.data_x, self.data_y):
                self.predictions_cache = [None, None]
                self.best_params = new_params
                self.fitness_value = new_fitness
                self.parameters = new_params.parameters

    def init_best_params(self):
        best = EVOLUTIONARY_UNIT(self.layers_size)
        best.parameters = self.parameters.copy()
        best.fitness(self.data_x, self.data_y)
        return best

    def init_params_standard(self):
        self.n = self.data_x.shape[0] if self.data_x is not None else None
        np.random.seed(1)
        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = \
                np.random.randn(self.layers_size[layer],
                                self.layers_size[layer - 1]) / np.sqrt(self.layers_size[layer - 1])
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))

    def initialize_parameters(self):
        # self.layers_size.insert(0, self.data_x.shape[1])
        self.n = self.data_x.shape[0] if self.data_x is not None else None

        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = np.random.normal(loc=0, scale=self.sigma, size=(self.layers_size[layer],self.layers_size[layer - 1]))
            # print("shape of W: ", self.parameters["W" + str(layer)].shape)
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))

        self.best_params = self.init_best_params()
        self.fitness_value = self.best_params.fitness_value

        return self

    def crossover(self, other, k, probBias):
        child = EVOLUTIONARY_UNIT(self.layers_size)
        child.parameters = self.parameters.copy()
        # for every kth component in self.parameters, replace it with the corresponding component in other.parameters

        for layer in range(1, len(self.layers_size)):
            for i in range(0, self.parameters["W" + str(layer)].shape[0]):
                for j in range(0, self.parameters["W" + str(layer)].shape[1]):
                    if ((i+1)*layer) % k == 0:
                        # generate random number between 0 and 1 with uniform distribution
                        rand = np.random.uniform(0, 1)
                        if rand < probBias:
                            child.parameters["W" + str(layer)][i][j] = self.parameters["W" + str(layer)][i][j]
                        else:
                            child.parameters["W" + str(layer)][i][j] = other.parameters["W" + str(layer)][i][j]

        return child




    # <editor-fold desc="Code for Prediction">
    def forward(self, data):
        store = {}
        # x.T is the transpose of x
        A = data.T

        # self.L is the number of layers
        for i in range(self.num_layers - 1):
            # print("W" + str(i + 1))
            try:
                Z = self.parameters["W" + str(i + 1)].dot(A) + self.parameters["b" + str(i + 1)]
            except KeyError:
                self.initialize_parameters()
                Z = self.parameters["W" + str(i + 1)].dot(A) + self.parameters["b" + str(i + 1)]
            A = self.sigmoid(Z)
            store["A" + str(i + 1)] = A
            store["W" + str(i + 1)] = self.parameters["W" + str(i + 1)]
            store["Z" + str(i + 1)] = Z


        Z = self.parameters["W" + str(self.num_layers)].dot(A) + self.parameters["b" + str(self.num_layers)]
        A = self.sigmoid(Z)
        store["A" + str(self.num_layers)] = A
        store["W" + str(self.num_layers)] = self.parameters["W" + str(self.num_layers)]
        store["Z" + str(self.num_layers)] = Z
        return A, store

    def sigmoid(self, Z):
        # clip z to -10, 10 to avoid overflow
        Z = np.clip(Z, -10, 10)
        return 1 / (1 + np.exp(-Z))

    def predict(self, data_x, target_out_y, test=False):
        if self.predictions_cache[1 if test else 0] is not None:
            return self.predictions_cache[1 if test else 0]
        # A is the output of the last layer
        A, cache = self.forward(data_x)
        number_examples = data_x.shape[0]
        predictions = np.zeros((1, number_examples))
        my_predictions = []
        for i in range(0, A.shape[1]):
            if A[0, i] > 0.5:
                predictions[0, i] = 1
                my_predictions.append(1)
            else:
                predictions[0, i] = 0
                my_predictions.append(0)
        predictions_Result = pd.DataFrame({'Actual': target_out_y, 'Predicted': my_predictions})
        self.predictions_cache[1 if test else 0] = predictions_Result
        return predictions_Result
    # </editor-fold>

    # <editor-fold desc="Accuracy_Fitness">
    def accuracy(self, predictions=None):
        # x_subset, y_subset = random_select(data_x, target_out_y, 100) if not test_set else (data_x, target_out_y)
        # print(self.data_y)
        predictions = self.predict(self.data_x, self.data_y) if predictions is None else predictions
        correct = [x for x in predictions.to_numpy().tolist() if x[0] == x[1]]
        accuracy = len(correct) / len(predictions)
        return accuracy

    def get_conf_table_values(self):
        predictions = self.predict(self.data_x, self.data_y)
        tp = predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 1)]
        tn = predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 0)]
        fp = predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 1)]
        fn = predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 0)]
        return len(tp), len(tn), len(fp), len(fn)

    def fitness(self, xdata, ydata, override=False):

        if self.fitness_value is not None and not override:
            return self.fitness_value
        else:
            self.data_x = xdata
            self.data_y = ydata

            self.fitness_value = self.accuracy()
            # print(self.fitness_value)
            return self.fitness_value

        predictions = self.predict(self.data_x, self.data_y)
        self.tp = len(predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 1)])
        self.tn = len(predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 0)])
        self.fp = len(predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 1)])
        self.fn = len(predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 0)])
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        # precision = (tp / (tp + fp))
        # recall = (tp / (tp + fn)) * (tn / (tn + fp))
        # f1 = 2 * (precision * recall) / (precision + recall)
        # I need to ensure that the
        return accuracy
    # </editor-fold>
