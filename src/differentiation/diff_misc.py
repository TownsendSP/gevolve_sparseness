import pickle

import numpy as np
from pandas import DataFrame
from tqdm.auto import tqdm

from src import analysis, parameter_stuff_and_things as param

def randChoiceNoDupes(arrlen, chosen, numTo):
    choices = np.random.choice(self.mu_indivs, 3, replace=False)
    while choices[0] == chosen or choices[1] == chosen or choices[2] == chosen:
        choices = np.random.choice(self.mu_indivs, 3, replace=False)
    return choices[0], choices[1], choices[2]
