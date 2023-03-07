import numpy as np
import pandas as pd


class EVOLUTIONARY_UNIT:
    def __init__(self, layers_size, data_x=None, data_y=None):
        self.parameters = {}
        self.layers_size = layers_size
        self.num_layers = len(layers_size)-1
        self.data_x = data_x
        self.data_y = data_y
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

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

    def update_params(self, sigma):
        new_params = self.parameters
        for layer in range(1, len(self.layers_size)):
            random_gaussian_array = np.random.normal(loc=0, scale=sigma, size=(self.layers_size[layer], self.layers_size[layer - 1]))
            spoon = np.add(new_params["W" + str(layer)], random_gaussian_array)
            spoon = np.clip(spoon, -10, 10)
            new_params["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))
            new_params["W" + str(layer)] = spoon
        self.parameters = new_params
        # append new params to ./params.txt
        with open("params.txt", "a") as f:
            f.write(str(new_params) + "\n")
        return new_params


    def generate_fucking_proper_random_gaussian_list(self, sugma, longness):
        mean = 0
        std = sugma

        return np.random.normal(mean, std, longness)



    # def update_params(self, sigma):
    #     new_params = self.parameters.copy()
    #     for layer in range(1, len(self.layers_size)):
    #         # Randomize weights
    #         new_weights = np.add(new_params["W" + str(layer)], np.random.normal(scale=sigma, size=(
    #         self.layers_size[layer], self.layers_size[layer - 1])))
    #         new_weights = np.clip(new_weights, -10, 10)
    #         new_params["W" + str(layer)] = new_weights
    #         # Randomize biases
    #         new_biases = np.add(new_params["b" + str(layer)],
    #                             np.random.normal(scale=sigma, size=(self.layers_size[layer], 1)))
    #         new_biases = np.clip(new_biases, -10, 10)
    #         new_params["b" + str(layer)] = new_biases
    #     self.parameters = new_params
    #     return new_params["W1"][0]


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

    def cost(self):
        A, store = self.forward(self.data_x)
        cost = np.squeeze(-(self.data_y.dot(np.log(A.T if A.T != 0 else 0.001)) + (1 - self.data_y).dot(np.log(1 - (A.T if A.T != 0 else 0.001)))) / self.n)

        return cost

    def get_conf_table_values(self):
        predictions = self.predict(self.data_x, self.data_y)
        tp = predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 1)]
        tn = predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 0)]
        fp = predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 1)]
        fn = predictions[(predictions['Actual'] == 1) & (predictions['Predicted'] == 0)]
        return len(tp), len(tn), len(fp), len(fn)

    def fitness(self):
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
