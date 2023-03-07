
from src import parameter_stuff_and_things as param
from src import processing as pro
import pickle

# %%
layers_dims = [16, 34, 1]
with open ('./data/model_0.pkl', 'rb') as f:
    model_final = pickle.load(f)
# %%
model_final_params = model_final

train_x, train_y, test_x, test_y = pro.split_data(pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df("./data/DryBeanDataset/Dry_Bean_Dataset.arff"))))

print("Final Test Accuracy: " + str(param.accuracy(test_x, test_y, model_final_params, layers_dims)) + "%")
predictions = param.predict_cheaty(test_x, test_y, model_final_params, layers_dims)

# %%
# params at w1
data = model_final["W1"]
flat_data = [abs(x) for row in data for x in row]
sorted_data = sorted(flat_data, reverse=True)
threshold = sorted_data[int(0.1 * len(sorted_data))]
data = [[1 if abs(x) >= threshold else 0 for x in row] for row in data]


data = model_final["W1"]
flat_data = [abs(x) for row in model_final["W1"] for x in row]
sorted_data = sorted([abs(x) for row in model_final["W1"] for x in row], reverse=True)
threshold = sorted([abs(x) for row in model_final["W1"] for x in row], reverse=True)[int(0.1 * len(sorted([abs(x) for row in model_final["W1"] for x in row], reverse=True)))]
ab = [[1 if abs(x) >= sorted([abs(x) for row in model_final["W1"] for x in row], reverse=True)[int(0.1 * len(sorted([abs(x) for row in model_final["W1"] for x in row], reverse=True)))] else 0 for x in row] for row in model_final["W1"]]
