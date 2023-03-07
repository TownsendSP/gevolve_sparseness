import os
import pickle
import shutil
from datetime import datetime

import pandas as pd

import src.genetics.evolutionary_genetics as gevo
from src import analysis as analysis, processing as pro
from src.backpropogation import backprop as bprop
from src.evolving import v1evolver as evo1
from src.genetics import genetic_postprocessing as gpost
from src.genetics import genetic_multiprocessor as gevo_multi


# %%
def load_csvs_to_df(startOfFileName):
    dfs = []
    for file in os.listdir("./megaRuns"):
        if file.startswith(startOfFileName):
            print(file)
            dfs.append(pd.read_csv("./megaRuns/" + file, header=0))
    return dfs

def backprop_train(beans, runs, iterations, layers_dims):
    print(layers_dims)
    backprop = bprop.BACKPROP(beans, runs, iterations, layers_dims)
    backprop.multiprocessor()
    print("Loading CSVs")
    training_accuracies = load_csvs_to_df("run_training")
    testing_accuracies = load_csvs_to_df("run_testing")
    training_average_df = analysis.split_and_recombine(training_accuracies)
    testing_average_df = analysis.split_and_recombine(testing_accuracies)
    training_average_df.to_csv("./megaRuns/average_training_accuracy.csv", index=False)
    testing_average_df.to_csv("./megaRuns/average_testing_accuracy.csv", index=False)
    return training_average_df, testing_average_df

def evo_v1_train(beans, runs, iterations, parent,sigma, spawn, mutation_rate, sampling_frequency, layers_dims):

    multiEvo = evo1.EVOLVER(runs, iterations, parent, sigma, spawn, mutation_rate, sampling_frequency, layers_dims, beans, 8)
    multiEvo.multiprocessor()

def setup_dirs():
    # delete all the run_n dirs in megaRuns
    for run in os.listdir("./megaRuns"):
        if run.startswith("runs_"):
            shutil.rmtree("./megaRuns/" + run)

    # delete files starting with "run_" in megaRuns
    for file in os.listdir("./megaRuns"):
        if file.startswith("run_"):
            os.remove("./megaRuns/" + file)


def final_analysis(number_of_runs):
    training_accuracies = load_csvs_to_df("run_training")
    testing_accuracies = load_csvs_to_df("run_testing")
    training_average_df = analysis.split_and_recombine(training_accuracies)
    testing_average_df = analysis.split_and_recombine(testing_accuracies)
    training_average_df.to_csv("./megaRuns/average_training_accuracy.csv", index=False)
    testing_average_df.to_csv("./megaRuns/average_testing_accuracy.csv", index=False)
    # # # </editor-fold>
    #
    test_accuracy_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv", header=0)
    train_accuracy_df = pd.read_csv("./megaRuns/average_training_accuracy.csv", header=0)
    # Graphing
    analysis.plot_graph_v4(train_accuracy_df,
                           test_accuracy_df,
                           number_of_runs,
                           "./megaRuns/Final_Graph_Dataset_comparison.png")
    analysis.load_avg_df_to_conf_mat(train_accuracy_df, "./megaRuns/training_conf_mat.png",
                                     "Training Data Confusion Matrix")
    analysis.load_avg_df_to_conf_mat(test_accuracy_df, "./megaRuns/testing_conf_mat.png",
                                     "Testing Data Confusion Matrix")
    outputFileName = ".\\output\\MegaRun_" + str(datetime.now()). \
        replace(" ", "_").replace(":", "-").replace(".", "-") + ".zip"
    os.system("zip -r " + outputFileName + " .\\megaRuns\\*")

def training_main():
    global output_subdirectory
    setup_dirs()
    beanpath = "./data/DryBeanDataset/Dry_Bean_Dataset.arff"
    beans = pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df(beanpath)))
    number_of_runs = 1
    num_iters = 1000
    indivs_per_gen = 16 * 34
    sigma = 0.5
    children_per_gen = 20
    # rate_of_mutation = 0.05
    rate_of_mutation = [0.05, 0.07, 0.1, 0.01]
    sampling_frequency = 5
    layers_dims = [34, 1]

    # evo_v1_train(beans, number_of_runs, num_iters, indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, layers_dims)
    # backprop_train(beans, number_of_runs, num_iters, layers_dims)

    final_analysis(number_of_runs)


def main():
    # %%
    global output_subdirectory
    setup_dirs()
    beanpath = "./data/DryBeanDataset/Dry_Bean_Dataset.arff"
    beans = pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df(beanpath)))
    source_params = pickle.load(open("./data/model_0.pkl", "rb"))
    number_of_runs = 10
    num_iters = 1100
    indivs_per_gen = 2
    # sigma = 0.5
    # children_per_gen = 20
    # rate_of_mutation = 0.05
    # rate_of_mutation = [0.05, 0.07, 0.1, 0.01]
    # sampling_frequency = 5
    layers_dims = [34, 1]
    naive_train = gpost.produce_naive_mask(source_params, beans.drop(['Class'], axis=1), beans['Class'], layers_dims, num_iters)

    gevolver = gevo_multi.EVOLVER(number_of_runs=number_of_runs, beans=beans, layers_dims=layers_dims, source_params=source_params, num_iters=num_iters, indivs_per_gen=indivs_per_gen, number_of_processes=8)
    gevolver.multiprocessor()

    # evo_v1_train(beans, number_of_runs, num_iters, indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, layers_dims)
    # %%
    # genetic = gevo.POPULATION(num_iters, indivs_per_gen, layers_dims, source_params)
    # genetic.data_x, genetic.data_y, genetic.test_data_x, genetic.test_data_y = pro.split_data(beans)
    # genetic.train_population()
    # trimmed = genetic.best_individual
    # trimmed.save("./runs/model_0.pkl")





    # %%


    # <editor-fold desc="Final_Analysis">
    training_accuracies = load_csvs_to_df("run_training")
    testing_accuracies = load_csvs_to_df("run_testing")
    training_average_df = analysis.split_and_recombine(training_accuracies)
    testing_average_df = analysis.split_and_recombine(testing_accuracies)
    training_average_df.to_csv("./megaRuns/average_training_accuracy.csv", index=False)
    testing_average_df.to_csv("./megaRuns/average_testing_accuracy.csv", index=False)

    #
    test_accuracy_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv", header=0)
    train_accuracy_df = pd.read_csv("./megaRuns/average_training_accuracy.csv", header=0)
    # Graphing
    analysis.plot_graph_comparison2(train_accuracy_df,
                           test_accuracy_df,
                           # naive_train,
                           number_of_runs,
                           "./megaRuns/Final_Graph_Dataset_comparison.png")
    analysis.plot_conf_comparison(train_accuracy_df,
                           test_accuracy_df,
                           naive_train,
                           "./megaRuns/Final_Confused_Dataset_comparison.png")
    # analysis.load_avg_df_to_conf_mat(train_accuracy_df, "./megaRuns/training_conf_mat.png",
    #                                  "Training Data Confusion Matrix")
    # analysis.load_avg_df_to_conf_mat(test_accuracy_df, "./megaRuns/testing_conf_mat.png",
    #                                  "Testing Data Confusion Matrix")
    outputFileName = ".\\output\\MegaRun_" + str(datetime.now()). \
        replace(" ", "_").replace(":", "-").replace(".", "-") + ".zip"
    os.system("zip -r " + outputFileName + " .\\megaRuns\\*")
    # </editor-fold>


# Press the green button in the gutter to run the script.
# %%
if __name__ == '__main__':
    output_subdirectory = "megaRuns"
    main()
    # beanpath = "./data/DryBeanDataset/Dry_Bean_Dataset.arff"
    # beans = pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df(beanpath)))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
