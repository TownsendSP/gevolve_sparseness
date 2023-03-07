import numpy as np
import pandas as pd
from src.evolving import ev_indiv as evi


def init_parameters_no_indiv_mutation(dataShape, layersSize, sigma=10):
    parameters = {}
    # print(str.format("Layers size: {0}", str(layersSize)))
    for layer in range(1, len(layersSize)):
        parameters["W" + str(layer)] = np.random.normal(loc=0, scale=sigma, size=(layersSize[layer], layersSize[layer - 1]))
        parameters["b" + str(layer)] = np.zeros((layersSize[layer], 1))
    return parameters

def init_parameters(dataShape, layers_size, sigma=10):
    parameters = {}
    # print(str.format("Layers size: {0}", str(layersSize)))
    for layer in range(1, len(layers_size)):
        parameters["W" + str(layer)] = np.random.normal(loc=0, scale=sigma if (type(sigma) == float) or (type(sigma) == int) else sigma[layer]
                                      , size=(layers_size[layer], layers_size[layer - 1]))
        parameters["b" + str(layer)] = np.zeros((layers_size[layer], 1))
    return parameters

def update_parameters(parameters, sigma, layers_size):
    new_params = parameters.copy()
    for layer in range(1, len(layers_size)):
        # print(sigma if (type(sigma) == float or type(sigma) == int) else sigma[layer])
        rand_gauss = np.random.normal(loc=0, scale=sigma if (type(sigma) == float or type(sigma) == int) else sigma[layer]
                                      , size=(layers_size[layer], layers_size[layer - 1]))
        new_params["W" + str(layer)] = np.add(new_params["W" + str(layer)], rand_gauss)
        new_params["b" + str(layer)] = np.zeros((layers_size[layer], 1))
    return new_params

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward(data, parameters, layers_size):
    store = {}
    A = data.T
    lenThing = len(layers_size) - 1

    for i in range(lenThing):
        layer = i + 1
        Z = parameters["W" + str(layer)].dot(A) + parameters["b" + str(layer)]
        A = sigmoid(Z)
        store["A" + str(layer)] = A
        store["W" + str(layer)] = parameters["W" + str(layer)]
        store["Z" + str(layer)] = Z

    print(str.format("lenThing: {0};data shape: {1};A shape: {2};layers_size: {3}", str(lenThing), str(data.shape), str(A.shape), str(layers_size)))

    Z = parameters["W" + str(lenThing)].dot(A) + parameters["b" + str(lenThing)]
    A = sigmoid(Z)
    store["A" + str(lenThing)] = A
    store["W" + str(lenThing)] = parameters["W" + str(lenThing)]
    store["Z" + str(lenThing)] = Z
    return A, store

def predict_old(data_x, target_out_y, parameters, layers_size):
    A, cache = forward(data_x, parameters, layers_size)
    number_examples = data_x.shape[0]
    predictions = np.zeros((1, number_examples))
    my_predictions = []
    for i in range(0, A.shape[1]):
        print(str.format("A[0, i]: {0}", str(A[0, i])))
        if A[0, i] > 0.5:
            predictions[0, i] = 1
            my_predictions.append(1)
        else:
            predictions[0, i] = 0
            my_predictions.append(0)
    return pd.DataFrame({'Actual': target_out_y, 'Predicted': my_predictions})

def predict_cheaty(data_x, target_out_y, parameters, layers_size):
    spork = evi.EVOLUTIONARY_UNIT(layers_size)
    spork.parameters = parameters
    return spork.predict(data_x, target_out_y)

def accuracy(data_x, target_out_y, parameters, layers_size):
    predictions = predict_cheaty(data_x, target_out_y, parameters, layers_size)
    return np.sum(predictions["Actual"] == predictions["Predicted"]) / len(predictions)

def fitness(data_x, target_out_y, parameters, layers_size):
    return accuracy(data_x, target_out_y, parameters, layers_size)