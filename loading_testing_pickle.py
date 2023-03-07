
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


# %%
layers_dims = [17, 34, 1]
with open ('./data/model_0.pkl', 'rb') as f:
    model_final = pickle.load(f)

train_x, train_y, test_x, test_y = pro.split_data(pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df("./data/DryBeanDataset/Dry_Bean_Dataset.arff"))))

print("Final Test Accuracy: " + str(param.accuracy(test_x, test_y, model_final, layers_dims)) + "%")
# print("Improvement: " + str(param.accuracy(test_x, test_y, model_final, model_final_ann.layers_size) - param.accuracy(test_x, test_y, model_initial, model_initial_ann.layers_size)) + "%")


