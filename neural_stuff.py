
import pickle
import parameter_stuff_and_things as param
import analysis
import os
import pickle
import queue  # imported for using queue.Empty exception
from datetime import datetime
from multiprocessing import Process, Queue

import evolutionary_superclass as evo
import numpy as np
import processing as pro
import parameter_stuff_and_things as param
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import analysis


class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.num_layers = len(self.layers_size)
        self.n = 0
        self.costs = []
        # creates dataframe for accuracy
        self.train_accuracy_df = pd.DataFrame(columns=['Iteration',
                                                 'Accuracy',
                                                 'True Positives',
                                                 'True Negatives',
                                                 'False Positives',
                                                 'False Negatives'])
        self.test_accuracy_df = pd.DataFrame(columns=['Iteration',
                                                 'Accuracy',
                                                 'True Positives',
                                                 'True Negatives',
                                                 'False Positives',
                                                 'False Negatives'])
        self.use_test_set = False
        self.test_set_x = None
        self.test_set_y = None
        self.output_subdir = "./runs/"

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def initialize_parameters(self):
        np.random.seed(1)
        for layer in range(1, len(self.layers_size)):
            self.parameters["W" + str(layer)] = \
                np.random.randn(self.layers_size[layer],
                                self.layers_size[layer - 1]) / np.sqrt(self.layers_size[layer - 1])
            self.parameters["b" + str(layer)] = np.zeros((self.layers_size[layer], 1))
        self.save_parameters()
        # print("Parameters:")
        # print(self.parameters)
        # print("end of parameters")
        # exit(0)

    def save_parameters(self):
        with open(self.output_subdir + 'initial_parameters.pkl', 'wb') as f:
            pickle.dump(self.parameters, f)

    def load_parameters(self, file_path):
        with open(file_path, 'rb') as f:
            self.parameters = pickle.load(f)

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

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def backward(self, data_x, target_out_y, store):
        derivatives = {}

        store["A0"] = data_x.T

        A = store["A" + str(self.num_layers)]
        dA = -np.divide(target_out_y, A) + np.divide(1 - target_out_y, 1 - A)

        dZ = dA * self.sigmoid_derivative(store["Z" + str(self.num_layers)])
        dW = dZ.dot(store["A" + str(self.num_layers - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.num_layers)].T.dot(dZ)

        derivatives["dW" + str(self.num_layers)] = dW
        derivatives["db" + str(self.num_layers)] = db

        for i in range(self.num_layers - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(i)])
            dW = 1. / self.n * dZ.dot(store["A" + str(i - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if i > 1:
                dAPrev = store["W" + str(i)].T.dot(dZ)

            derivatives["dW" + str(i)] = dW
            derivatives["db" + str(i)] = db

        return derivatives

    def fit(self, data_x, target_out_y, learning_rate=0.01, n_iterations=2500):
        np.random.seed(1)

        self.n = data_x.shape[0]

        self.layers_size.insert(0, data_x.shape[1])

        self.initialize_parameters()
        old_parameters = self.parameters
        new_parameters = self.parameters

        for loop in tqdm(range(n_iterations)):
            A, store = self.forward(data_x)

            cost = np.squeeze(-(target_out_y.dot(np.log(A.T)) + (1 - target_out_y).dot(np.log(1 - A.T))) / self.n)
            derivatives = self.backward(data_x, target_out_y, store)

            for l in range(1, self.num_layers + 1):
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]

            self.costs.append(cost)
            if 1==1:
                accuracy = analysis.test_accuracy(self, data_x, target_out_y, loop)
                if self.use_test_set and self.test_set_x is not None and self.test_set_y is not None:
                    analysis.test_accuracy(self, self.test_set_x, self.test_set_y, loop, test_set=True)
                # trim accuracy to 2 decimal places

                filename = str.format("parameters_{0}_{1}.pkl", loop, str(accuracy)[:4])
                with open('runs/' + filename, 'wb') as f:
                    pickle.dump(self, f)

                # find difference between old and new parameters
                strParams = str(self.parameters)
                # write to file
                with open('runs/params.txt', 'a') as f:
                    f.write(strParams + "")

                print("accuracy: " + str(accuracy)[:4] + " %")

                with open('runs/' + filename, 'rb') as f:
                    model_final_ann = pickle.load(f)

                print("Initial Test Accuracy: " + str(param.accuracy(model_final_ann.test_set_x, model_final_ann.test_set_y, model_final_ann.parameters, model_final_ann.layers_size)))


                # print("saved parameters to file: " + filename)

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

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.title("Learning rate")
        plt.savefig(self.output_subdir + "cost_graph.png")
        # plt.show()

