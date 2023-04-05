import numpy as np
import pandas as pd


class EVOLUTIONARY_UNIT:
    def __init__(self, layers_size, data_x=None, data_y=None):
        self.parameters = {}
        self.layers_size = layers_size
        self.num_layers = len(layers_size)-1
        self.data_x = data_x
        self.data_y = data_y
        self.fitness_value = None

        self.n = self.data_x.shape[0] if self.data_x is not None else None

    def __str__(self):
        return str.format("layers_size: {0}\nnum_layers: {1}\nparameters shape: {2}", self.layers_size, self.num_layers, (self.parameters["W1"]).shape)

    def initialize_parameters(self):
        self.n = self.data_x.shape[0] if self.data_x is not None else None
        np.random.seed(1)
        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = \
                np.random.randn(self.layers_size[layer],
                                self.layers_size[layer - 1]) / np.sqrt(self.layers_size[layer - 1])
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))

    def init_params_v2(self, sigma=0.01):
        # self.layers_size.insert(0, self.data_x.shape[1])
        self.n = self.data_x.shape[0] if self.data_x is not None else None

        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = np.random.normal(loc=0, scale=sigma, size=(self.layers_size[layer],self.layers_size[layer - 1]))
            # print("shape of W: ", self.parameters["W" + str(layer)].shape)
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))

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

    # <editor-fold desc="MathStuff">
    def diff(self, other):
    #     calculates vector difference between self and other
        difference = EVOLUTIONARY_UNIT(self.layers_size)
        difference.parameters = self.parameters.copy()
        for layer in range(1, len(self.layers_size)):
            difference.parameters["W" + str(layer)] -= other.parameters["W" + str(layer)]
        return difference


    def mul(self, factor):
    #     multiplies every parameter in self by factor, returns new object
        product = EVOLUTIONARY_UNIT(self.layers_size)
        product.parameters = self.parameters.copy()
        for layer in range(1, len(self.layers_size)):
            product.parameters["W" + str(layer)] *= factor
        return product



    def add(self, other):
    #     adds every parameter in self by factor, returns new object
        sum = EVOLUTIONARY_UNIT(self.layers_size)
        sum.parameters = self.parameters.copy()
        for layer in range(1, len(self.layers_size)):
            sum.parameters["W" + str(layer)] += other.parameters["W" + str(layer)]
        return sum
    # </editor-fold>


    # <editor-fold desc="Code for Prediction">
    def forward(self, data):
        store = {}
        # x.T is the transpose of x
        A = data.T

        # self.L is the number of layers
        for i in range(self.num_layers - 1):
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

    def predict(self, data_x, target_out_y):
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
        return predictions_Result
    # </editor-fold>

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

    def fitness(self, xdata = None, ydata = None):
        if xdata is not None and ydata is not None:
            self.data_x = xdata
            self.data_y = ydata

        if self.fitness_value is not None:
            return self.fitness_value
        else:

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
