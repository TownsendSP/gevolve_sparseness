import os
import pickle
import queue  # imported for using queue.Empty exception
from datetime import datetime
from multiprocessing import Process, Queue

from src.evolving import evolutionary_superclass as evo
import numpy as np
from src import processing as pro


def train(mu, sigma, lambda_childs, r, samp_freq, beans, iterations, layers_dims, multi=True):
    train_x, train_y, test_x, test_y = pro.split_data(beans)

    if multi:
        output_subdirectory_train = "./megaRuns/runs/"
    else:
        output_subdirectory_train = "./runs/"

    model = evo.POPULATION(iterations, mu, sigma, lambda_childs, r, samp_freq, layers_dims, ind_mutations=True)
    model.data_x = train_x
    model.data_y = train_y
    model.test_data_x = test_x
    model.test_data_y = test_y
    teachers_pet = model.train_population()
    fitOverTime = model.fitOverTime

    npThingSpork = np.array(fitOverTime)
    np.savetxt(output_subdirectory_train + "fitOverTime.csv", npThingSpork, delimiter=",")



    # with open(output_subdirectory_train + 'model.pkl', 'wb') as f:
    #     pickle.dump(teachers_pet, f)

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
    return model.train_accuracy_df, model.test_accuracy_df, teachers_pet


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

                os.rename("../../megaRuns/runs/", "./megaRuns/runs_" + str(task))
                if not os.path.exists("../../megaRuns/runs"):
                    os.mkdir("../../megaRuns/runs")
                training_df.to_csv("./megaRuns/run_training" + str(task) + "_accuracy.csv", index=False)
                testing_df.to_csv("./megaRuns/run_testing" + str(task) + "_accuracy.csv", index=False)
                pickle.dump(neural_net, open("../../runs/run_model.pkl", "wb"))
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