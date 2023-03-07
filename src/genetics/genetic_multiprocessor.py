from src.genetics import genetic_postprocessing as gpost
import os
import pickle
import queue  # imported for using queue.Empty exception
from datetime import datetime
from multiprocessing import Process, Queue

from src.evolving import evolutionary_superclass as evo
import numpy as np
from src import processing as pro
import os
import pickle
import shutil
from datetime import datetime
import pandas as pd
from src import analysis as analysis, processing as pro
from src.backpropogation import backprop as bprop
from src.evolving import v1evolver as evo1
import src.genetics.evolutionary_genetics as gevo


def train(beans, runs, iterations, layers_dims, mu, source_params):
    train_x, train_y, test_x, test_y = pro.split_data(beans)
    output_subdirectory_train = "./megaRuns/runs/"

    genetic = gevo.POPULATION(iterations, mu, layers_dims, source_params)
    genetic.data_x, genetic.data_y, genetic.test_data_x, genetic.test_data_y = pro.split_data(beans)
    genetic.train_population()
    pickle.dump(genetic.best_individual, open(output_subdirectory_train + "best_individual_run_" + str(runs) + ".pkl", "wb"))
    pickle.dump(genetic.best_history, open(output_subdirectory_train + "best_history_run_" + str(runs) + ".pkl", "wb"))
    pickle.dump(genetic, open(output_subdirectory_train + "genetic_run_" + str(runs) + ".pkl", "wb"))
    train_accuracy_df = gpost.postprocess(genetic.best_history, source_params, genetic.data_x, genetic.data_y, layers_dims)
    test_accuracy_df = gpost.postprocess(genetic.best_history, source_params, genetic.test_data_x, genetic.test_data_y, layers_dims)

    train_accuracy_df.to_csv(output_subdirectory_train + "train_accuracy.csv", index=False)
    test_accuracy_df.to_csv(output_subdirectory_train + "test_accuracy.csv", index=False)


    return train_accuracy_df, test_accuracy_df, genetic.best_individual


class EVOLVER:
    def __init__(self, number_of_runs, num_iters, indivs_per_gen, source_params, layers_dims, beans, number_of_processes):
        self.number_of_runs = number_of_runs
        self.num_iters = num_iters
        self.indivs_per_gen = indivs_per_gen
        self.source_params = source_params
        self.layers_dims = layers_dims
        self.df = beans
        self.number_of_processes = number_of_processes

    def do_job(self, tasks, results):
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                break
            else:
                # print('\nRun ' + task + ' is done by ' + current_process().name)
                training_df, testing_df, neural_net = train(self.df, task, self.num_iters, self.layers_dims, self.indivs_per_gen, self.source_params)
                # print("Training Complete!")
                # training_df.to_csv("./megaRuns/training" + str(task) + "_accuracy.csv", index=False)
                # testing_df.to_csv("./megaRuns/testing" + str(task) + "_accuracy.csv", index=False)
                # pickle.dump(neural_net, open("./megaRuns/model" + str(task) + ".pkl", "wb"))
                # self.models.append(neural_net)
                # self.training_accuracies[task] = training_df
                # self.testing_accuracies.append(testing_df)

                os.rename("./megaRuns/runs/", "./megaRuns/runs_" + str(task))
                if not os.path.exists("./megaRuns/runs"):
                    os.mkdir("./megaRuns/runs")
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