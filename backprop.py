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
import processing as pro

def train_backprop_beans(beans, iterations, layers_dims, multi=False):
    train_x, train_y, test_x, test_y = pro.split_data(beans)
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

class BACKPROP:
    def __init__(self, beans, runs, iterations, layers_dims):
        self.number_of_runs = runs
        self.num_iters = iterations

        self.layers_dims = layers_dims
        self.df = beans
        self.number_of_processes = 8

    def do_job(self, tasks, results):
        while True:
            try:
                task = tasks.get_nowait()
            except queue.Empty:
                break
            else:
                # print('\nRun ' + task + ' is done by ' + current_process().name)
                training_df, testing_df, neural_net = train_backprop_beans(self.df, self.num_iters, self.layers_dims, True)
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
