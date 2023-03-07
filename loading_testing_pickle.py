
from src import parameter_stuff_and_things as param
from src import processing as pro
import pickle

# %%
layers_dims = [16, 34, 1]
with open ('./data/model_0.pkl', 'rb') as f:
    model_final = pickle.load(f)

train_x, train_y, test_x, test_y = pro.split_data(pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df("./data/DryBeanDataset/Dry_Bean_Dataset.arff"))))

print("Final Test Accuracy: " + str(param.accuracy(test_x, test_y, model_final, layers_dims)) + "%")

# %%
# params at w1
print(model_final["W1"])



