import os
import pickle
import shutil
from datetime import datetime

import pandas as pd

import analysis as analysis
import backprop as bprop
import processing as pro
import v1evolver as evo1


def evoTrainThing2(indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, df, num_iters,layers_dims, multi=True):
    # print('\nRun ' + task + ' is done by ' + current_process().name)
    training_df, testing_df, neural_net = evo1.train(indivs_per_gen, sigma, children_per_gen,
                                                rate_of_mutation,
                                                sampling_frequency, df, num_iters,
                                                layers_dims, multi=True)
    # print("Training Complete!")
    # training_df.to_csv("./megaRuns/training" + str(task) + "_accuracy.csv", index=False)
    # testing_df.to_csv("./megaRuns/testing" + str(task) + "_accuracy.csv", index=False)
    # pickle.dump(neural_net, open("./megaRuns/model" + str(task) + ".pkl", "wb"))
    # self.models.append(neural_net)
    # self.training_accuracies[task] = training_df
    # self.testing_accuracies.append(testing_df)
    training_df.to_csv("./runs/run_training_accuracy.csv", index=False)
    testing_df.to_csv("./runs/run_testing_accuracy.csv", index=False)

    pickle.dump(neural_net, open("./runs/model.pkl", "wb"))
    return training_df, testing_df

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
    # %%
    # delete all the run_n dirs in megaRuns
    for run in os.listdir("./megaRuns"):
        if run.startswith("runs_"):
            shutil.rmtree("./megaRuns/" + run)

    # delete files starting with "run_" in megaRuns
    for file in os.listdir("./megaRuns"):
        if file.startswith("run_"):
            os.remove("./megaRuns/" + file)
    # %%

def main():
    global output_subdirectory
    setup_dirs()
    beanpath = "./data/DryBeanDataset/Dry_Bean_Dataset.arff"
    beans = pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df(beanpath)))
    number_of_runs = 10
    num_iters = 200
    indivs_per_gen = 4
    sigma = 0.5
    children_per_gen = 20
    # rate_of_mutation = 0.05
    rate_of_mutation = [0.05, 0.07, 0.1, 0.01]
    sampling_frequency = 5
    layers_dims = [34, 1]

    # train_accuracy_df, test_accuracy_df = evoTrainThing2(indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, beans, num_iters, layers_dims)
    evo_v1_train(beans, number_of_runs, num_iters, indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, layers_dims)
    # backprop_train(beans, number_of_runs, num_iters, layers_dims)
    # train_backprop_beans(beans, num_iters, layers_dims, multi=False)
    # train_accuracy_df, test_accuracy_df, ann =
    #
    #
    # (beans, num_iters, layers_dims, multi=False)


    # # <editor-fold desc="Backprop">
    #
    # print("Loading CSVs")
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    output_subdirectory = "megaRuns"
    main()
    # beanpath = "./data/DryBeanDataset/Dry_Bean_Dataset.arff"
    # beans = pro.scale_replace_beans(pro.normalise_beans(pro.arff_to_df(beanpath)))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
