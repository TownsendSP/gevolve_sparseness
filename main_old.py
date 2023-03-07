import json
import os
import pickle
import queue  # imported for using queue.Empty exception
import shutil
from datetime import datetime
from multiprocessing import Process, Queue

import pandas
import pandas as pd

import analysis as analysis
import evolutionary_superclass as evo
import neural_stuff as neural



def read_datasets():
    # %%
    dataframe_imported = pd.read_csv("../Project_2/data/dataset_31_credit-g.csv", header=0)
    train = dataframe_imported.sample(frac=0.7, random_state=200)
    test = dataframe_imported.drop(train.index)
    train_y = train["'class'"]
    train_x = train.drop(["'class'"], axis=1)
    test_y = test["'class'"]
    test_x = test.drop(["'class'"], axis=1)
    print(test_x)
    return train_x, train_y, test_x, test_y


def splitForTraining(df):
    good = df[df["'class'"] == 1]
    bad = df[df["'class'"] == 0]

    train_good = good.sample(frac=0.7, random_state=200)
    test_good = good.drop(train_good.index)
    train_bad = bad.sample(frac=0.7, random_state=200)
    test_bad = bad.drop(train_bad.index)

    train = pd.concat([train_good, train_bad])
    test = pd.concat([test_good, test_bad])

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    train.to_csv('./megaRuns/runs/train.csv', index=False)
    test.to_csv('./megaRuns/runs/test.csv', index=False)

    train_y = train["'class'"]
    train_x = train.drop(["'class'"], axis=1)
    test_y = test["'class'"]
    test_x = test.drop(["'class'"], axis=1)

    return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()


def split_v1(unsplit_dataframe):
    # splitting
    # good_results = unsplit_dataframe[unsplit_dataframe["'class'"] == 1]
    # bad_results = unsplit_dataframe[unsplit_dataframe["'class'"] == 0]
    # train_bad = bad_results.sample(frac=0.7, random_state=200)
    # test_bad = bad_results.drop(train_bad.index)
    # train_good = good_results.sample(frac=0.7, random_state=200)
    # test_good = good_results.drop(train_good.index)
    # train = pandas.concat([train_bad, train_good])
    # test = pandas.concat([test_bad, test_good])

    train = unsplit_dataframe.sample(frac=0.7, random_state=200)
    test = unsplit_dataframe.drop(train.index)

    # reset indices
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_y = train["'class'"]
    train_x = train.drop(["'class'"], axis=1)
    test_y = test["'class'"]
    test_x = test.drop(["'class'"], axis=1)
    # dataframes are weird, so I'm making it a list
    return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()


def normalize_data(dataframe):
    output_subdirectory = "./megaRuns/runs/"
    dataframe_copy = dataframe.copy()
    replacement_list = ["'checking_status'",
                        "'credit_history'",
                        "'purpose'",
                        "'savings_status'",
                        "'employment'",
                        "'personal_status'",
                        "'other_parties'",
                        "'property_magnitude'",
                        "'other_payment_plans'",
                        "'housing'",
                        "'job'",
                        "'own_telephone'",
                        "'foreign_worker'"]
    dataframe_copy['\'class\''].replace('good', 1, inplace=True)
    # replace all instances of 'good' in column 'class' with 1
    dataframe_copy['\'class\''].replace('bad', 0, inplace=True)
    # replace all instances of 'bad' in column 'class' with 0
    # duplicate all "bad" rows, and 100 of the randomly selected "bad" rows
    dataframe_copy = pandas.concat([dataframe_copy, dataframe_copy[dataframe_copy['\'class\''] == 0]])
    dataframe_copy = pandas.concat(
        [dataframe_copy, dataframe_copy[dataframe_copy['\'class\''] == 0].sample(n=100, random_state=1)])

    # shuffle the dataframe
    dataframe_copy = dataframe_copy.sample(frac=1).reset_index(drop=True)
    replacedList = []
    for i in replacement_list:
        # print(df[i].value_counts())
        uniques = dataframe_copy[i].unique()
        currCol = []
        for j in uniques:
            currCol.append([j, uniques.tolist().index(j)])
            dataframe_copy[i].replace(j, uniques.tolist().index(j), inplace=True)
        replacedList.append([i, currCol])

    for (column, columnData) in dataframe_copy.items():
        dataframe_copy[column] -= columnData.min()
        dataframe_copy[column] /= (columnData.max())
    dataframe_copy.to_csv(output_subdirectory + "normalized.csv", index=False)
    return dataframe_copy, json.dumps(replacedList)


def pre_process_data(unProcessed_dataframe):
    for (column, columnData) in unProcessed_dataframe.items():
        unProcessed_dataframe[column] -= columnData.min()
        unProcessed_dataframe[column] /= (columnData.max())
    return unProcessed_dataframe


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


def train_backprop(initial_dataframe, iterations, layers_dims, multi=False):
    initial_dataframe, replacedList = normalize_data(initial_dataframe)
    train_x, train_y, test_x, test_y = splitForTraining(initial_dataframe)
    # print("Training set: " + str(train_x.shape))
    # print("Test set: " + str(test_x.shape))
    if multi:
        output_subdirectory = "./megaRuns/runs/"
    else:
        output_subdirectory = "./runs/"

    ann = neural.ANN(layers_dims)
    ann.use_test_set = True
    ann.test_set_x = test_x
    ann.test_set_y = test_y
    ann.fit(train_x, train_y, learning_rate=0.15, n_iterations=iterations)
    ann.output_subdir = output_subdirectory
    ann.predict(train_x, train_y)
    ann.predict(test_x, test_y)

    with open(output_subdirectory + 'model.pkl', 'wb') as f:
        pickle.dump(ann, f)

    ann.train_accuracy_df.to_csv(output_subdirectory + "accuracy.csv", index=False)
    ann.test_accuracy_df.to_csv(output_subdirectory + "test_set_accuracy.csv", index=False)

    # save the model
    # create a zip file of folder runs using os.system
    if not multi:
        outputFileName = ".\\output\\run_" + str(datetime.now()). \
            replace(" ", "_").replace(":", "-").replace(".", "-") + ".zip"
        os.system("zip -r " + outputFileName + " .\\runs\\*")
    # print("Training_accuracy: " + str(ann.train_accuracy_df.shape))
    # print("Testing_accuracy: " + str(ann.test_accuracy_df.shape))
    return ann.train_accuracy_df, ann.test_accuracy_df, ann


def train(mu, sigma, lambda_childs, r, samp_freq, initial_dataframe, iterations, layers_dims, multi=False):
    # global output_subdirectory

    initial_dataframe, replacedList = normalize_data(initial_dataframe)
    train_x, train_y, test_x, test_y = splitForTraining(initial_dataframe)
    # print("Training set: " + str(train_x.shape))
    # print("Test set: " + str(test_x.shape))
    if multi:
        output_subdirectory_train = "./megaRuns/runs/"
    else:
        output_subdirectory_train = "./runs/"

    model = evo.POPULATION(iterations, mu, sigma, lambda_childs, r, samp_freq, layers_dims)
    model.data_x = train_x
    model.data_y = train_y
    model.test_data_x = test_x
    model.test_data_y = test_y
    model.train_population()

    with open(output_subdirectory_train + 'model.pkl', 'wb') as f:
        pickle.dump(model, f)

    model.train_accuracy_df.to_csv(output_subdirectory_train + "accuracy.csv", index=False)
    model.test_accuracy_df.to_csv(output_subdirectory_train + "test_set_accuracy.csv", index=False)

    # save the model
    # create a zip file of folder runs using os.system
    if not multi:
        outputFileName = ".\\output\\run_" + str(datetime.now()). \
            replace(" ", "_").replace(":", "-").replace(".", "-") + ".zip"
        os.system("zip -r " + outputFileName + " .\\runs\\*")
    # print("Training_accuracy: " + str(model.train_accuracy_df.shape))
    # print("Testing_accuracy: " + str(model.test_accuracy_df.shape))
    return model.train_accuracy_df, model.test_accuracy_df, model


class EVOLVER:
    def __init__(self, number_of_runs, num_iters, indivs_per_gen, sigma, children_per_gen, rate_of_mutation,
                 sampling_frequency, layers_dims, df, number_of_processes):
        self.number_of_runs = number_of_runs
        self.num_iters = num_iters
        self.indivs_per_gen = indivs_per_gen
        self.sigma = sigma
        self.children_per_gen = children_per_gen
        self.rate_of_mutation = rate_of_mutation
        self.sampling_frequency = sampling_frequency
        self.layers_dims = layers_dims
        self.df = df
        self.number_of_processes = number_of_processes

    def do_job(self, tasks, results):
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                break
            else:
                # print('\nRun ' + task + ' is done by ' + current_process().name)
                training_df, testing_df, neural_net = train(self.indivs_per_gen, self.sigma, self.children_per_gen,
                                                            self.rate_of_mutation,
                                                            self.sampling_frequency, self.df, self.num_iters,
                                                            self.layers_dims, multi=True)
                # print("Training Complete!")
                # training_df.to_csv("./megaRuns/training" + str(task) + "_accuracy.csv", index=False)
                # testing_df.to_csv("./megaRuns/testing" + str(task) + "_accuracy.csv", index=False)
                # pickle.dump(neural_net, open("./megaRuns/model" + str(task) + ".pkl", "wb"))
                # self.models.append(neural_net)
                # self.training_accuracies[task] = training_df
                # self.testing_accuracies.append(testing_df)
                os.rename("../Project_2/megaRuns/runs/", "./megaRuns/runs_" + str(task))
                if not os.path.exists("../Project_2/megaRuns/runs"):
                    os.mkdir("../Project_2/megaRuns/runs")
                training_df.to_csv("./megaRuns/run_training" + str(task) + "_accuracy.csv", index=False)
                testing_df.to_csv("./megaRuns/run_testing" + str(task) + "_accuracy.csv", index=False)

                pickle.dump(neural_net, open("./megaRuns/runs_" + str(task) + "/model_" + str(task) + ".pkl", "wb"))
                results.put(True)

        return True

    def multiprocessor(self):
        processes = []
        tasks = Queue()
        results = Queue()
        for i in range(self.number_of_runs):
            tasks.put(str(i))

        for w in range(self.number_of_processes):
            p = Process(target=self.do_job, args=(tasks, results))
            processes.append(p)
            p.start()
        # print("Processes Started!")

        for p in processes:
            # print("Joining process: " + str(p))
            p.join()
        print("Processes Joined!")

        # while not results.empty():
        #     print(str.format("Results Queue size: {0}\n", results.qsize()))
        #     results.
        # thing = results.get()
        #     self.training_accuracies.append(thing[0])
        #     self.testing_accuracies.append(thing[1])
        #     self.models.append(thing[2])
        # print(str(self.training_accuracies[0].shape))
        # self.training_accuracies[0].to_csv('./megaRuns/0_accuracy.csv', index=False)
        return True  # self.training_accuracies, self.testing_accuracies, self.models


def load_csvs_to_df(startOfFileName):
    dfs = []
    for file in os.listdir("../Project_2/megaRuns"):
        if file.startswith(startOfFileName):
            print(file)
            dfs.append(pd.read_csv("./megaRuns/" + file, header=0))
    return dfs


def main():
    global output_subdirectory
    setup_dirs()
    number_of_runs = 2
    num_iters = 100
    indivs_per_gen = 4
    # %%
    sigma = 0.2
    children_per_gen = 20
    rate_of_mutation = 0.1
    sampling_frequency = 100
    layers_dims = [40, 1]
    df = pd.read_csv("../Project_2/data/dataset_31_credit-g.csv", header=0)
    output_subdirectory = "./megaRuns/runs/"
    if not os.path.exists("../Project_2/megaRuns/runs"):
        os.mkdir("../Project_2/megaRuns/runs")

    # Main loop that automates the running and training of the ann
    multiEvo = EVOLVER(number_of_runs, num_iters, indivs_per_gen, sigma, children_per_gen, rate_of_mutation,
                       sampling_frequency, layers_dims, df, 10)
    # training_accuracies, testing_accuracies, models = multiEvo.multiprocessor()
    multiEvo.multiprocessor()
    # load all files that start with "run_training" into a list of dataframes
    # %%
    print("Loading CSVs")
    training_accuracies = load_csvs_to_df("run_training")
    testing_accuracies = load_csvs_to_df("run_testing")
    # train_data = pd.read_csv("./megaRuns/runs_0/train.csv")
    # print number of ones in 'class' column
    # print(train_data["'class'"].value_counts())
    # train_data = pd.read_csv("./megaRuns/runs_1/train.csv")
    # print number of ones in 'class' column
    # print(train_data["'class'"].value_counts())
    # print()
    # %%
    # print(models)
    # training_accuracies[0].to_csv(output_subdirectory + "testasdfaccuracy.csv", index=False)

    # training_accuracies = multiEvo.training_accuracies
    # testing_accuracies = multiEvo.testing_accuracies

    # for i in tqdm(range(number_of_runs)):
    #     training_df, testing_df, neural_net = train(indivs_per_gen, sigma, children_per_gen, rate_of_mutation, sampling_frequency, df, num_iters, layers_dims, multi=True)
    #     training_accuracies.append(training_df)
    #     testing_accuracies.append(testing_df)
    #     os.rename("./megaRuns/runs/", "./megaRuns/runs_" + str(i))
    #     if not os.path.exists("./megaRuns/runs"):
    #         os.mkdir("./megaRuns/runs")

    # Averaging all the dataframes from all the runs
    training_average_df = analysis.split_and_recombine(training_accuracies)
    testing_average_df = analysis.split_and_recombine(testing_accuracies)

    # saving all the dataframes to csv
    training_average_df.to_csv("./megaRuns/average_training_accuracy.csv", index=False)
    testing_average_df.to_csv("./megaRuns/average_testing_accuracy.csv", index=False)
    # %%
    testing_average_df = pd.read_csv("../Project_2/megaRuns/average_testing_accuracy.csv", header=0)
    training_average_df = pd.read_csv("../Project_2/megaRuns/average_training_accuracy.csv", header=0)

    # Graphing
    outputFileName = ".\\output\\MegaRun_" + str(datetime.now()). \
        replace(" ", "_").replace(":", "-").replace(".", "-") + ".zip"
    analysis.make_good_graph(training_average_df,
                             testing_average_df,
                             number_of_runs,
                             "megaRuns/Final_Graph.png")
    # %%
    # Making the confusion_matrices:
    analysis.load_avg_df_to_conf_mat(training_average_df, "./megaRuns/training_conf_mat.png",
                                     "Training Data Confusion Matrix")
    analysis.load_avg_df_to_conf_mat(testing_average_df, "./megaRuns/testing_conf_mat.png",
                                     "Testing Data Confusion Matrix")

    os.system("zip -r " + outputFileName + " .\\megaRuns\\*")


if __name__ == '__main__':
    main()
