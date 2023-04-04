import pandas as pd

from src import parameter_stuff_and_things as param

def full_to_list(nested_ndarray):
    return [[y.tolist() for y in x] for x in nested_ndarray]


def apply_mask(gene, params):
    brain = params.copy()
    brain["W1"] = params["W1"] * gene
    return brain

def postprocess(gene_array, params, data_x, data_y, layers_dims):
    brains = [apply_mask(gene_array[i], params) for i in range(len(gene_array))]
    # print(len(brains))
    # accuracies = [param.accuracy(data_x, data_y, brains[i], layers_dims) for i in range(len(gene_array))]
    predictions = [param.predict_cheaty(data_x, data_y, brains[j], [16] + layers_dims) for j in range(len(brains))]
#     create dataframe of i, accuracy pairs
    positivies, negatives = len([x for x in data_y.tolist() if x == 1]), len([x for x in data_y.tolist() if x == 0])

    df = pd.DataFrame({"Iteration": [i for i in range(len(gene_array))],
                        "Accuracy": [len([x for x in preds.to_numpy().tolist() if x[0] == x[1]])/len(preds) for preds in predictions],
                        "True Positives": [len([x for x in preds.to_numpy().tolist() if x[0] == x[1] and x[0] == 1])for preds in predictions],
                        "True Negatives": [len([x for x in preds.to_numpy().tolist() if x[0] == x[1] and x[0] == 0])for preds in predictions],
                        "False Positives": [len([x for x in preds.to_numpy().tolist() if x[0] != x[1] and x[0] == 1])for preds in predictions],
                        "False Negatives": [len([x for x in preds.to_numpy().tolist() if x[0] != x[1] and x[0] == 0])for preds in predictions],
                        "Positives": [len([x for x in data_y.tolist() if x == 1]) for i in range(len(gene_array))],
                        "Negatives": [len([x for x in data_y.tolist() if x == 0]) for i in range(len(gene_array))]})
    return df

def naive(model_final):
    data = model_final["W1"]
    flat_data = [abs(x) for row in data for x in row]
    sorted_data = sorted(flat_data, reverse=True)
    threshold = sorted_data[int(0.1 * len(sorted_data))]
    data = [[1 if abs(x) >= threshold else 0 for x in row] for row in data]
    return data

def produce_naive_mask(model_final, data_x, data_y, layers_dims, iterations):

    gene_array = [naive(model_final) for i in range(iterations)]
    # print(len(gene_array))
    return postprocess(gene_array, model_final, data_x, data_y, [34, 1])



