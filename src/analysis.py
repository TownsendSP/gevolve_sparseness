import os

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

from src import parameter_stuff_and_things as param


def hist_of_freq_of_df_column(df, col_name):
    df[col_name].value_counts().plot(kind='bar')
    # make sure titles fully on graph
    plt.tight_layout()
    plt.show()


def boxplot_of_df(title, df):
    df.boxplot()
    plt.title(title)

    plt.show()


# def bean_dataset_preparation():


def box_plots_of_df_title(df_list):
    for title, df in df_list:
        boxplot_of_df(title, df)


# use Pandas to read csv file into a dataframe
def readCsvToDataFrame(filename):
    df = pd.read_csv(filename, header=0)
    return df


# create a histogram of each column
def createHistogram(df):
    numAttrs = len(df.columns)
    print("Number of attributes: ", numAttrs)
    for column in df:
        plt.hist(df[column])
        plt.title(column)
        # plt.show()


# create a histogram of each column, with df1 in red and df2 in blue
def createHistogram2(df1, df2, df3):
    numAttrs = len(df1.columns)
    print("Number of attributes: ", numAttrs)
    for column in df1:
        plt.hist(df1[column], color='blue')
        plt.hist(df2[column], color='red')
        plt.hist(df3[column], color='green')
        plt.title(column)
        # plt.show()


# create a histogram of each column, with df1 in red and df2 in blue
def createAndSaveHistogram2(df1, df2, df3, outputDir):
    numAttrs = len(df1.columns)
    print("Number of attributes: ", numAttrs)
    os.mkdir(outputDir + "/histograms")
    for column in df1:
        plt.hist(df1[column], color='blue')
        plt.hist(df2[column], color='red')
        plt.hist(df3[column], color='green')
        plt.title(column)
        plt.savefig(outputDir + "/histograms/" + column + ".png")


def calcConfusionMatNums(df):
    trues = df[df['Actual'] == df['Predicted']]
    true_positives = trues[trues['Actual'] == 1]
    true_negatives = trues.drop(true_positives.index)

    falses = df.drop(trues.index)
    false_positives_confused = falses[falses['Predicted'] == 1]
    false_negatives = falses.drop(false_positives_confused.index)
    # y_test = (df['Actual']).to_numpy()
    # y_pred = (df['Predicted']).to_numpy()

    return len(true_positives), len(true_negatives), len(false_positives_confused), len(false_negatives)


def display_confused_mat(df, title, outputDir="./runs"):
    actual_trues = df[df['Actual'] == 1]
    actual_falses = df[df['Actual'] == 0]
    predicted_trues = df[df['Predicted'] == 1]
    predicted_falses = df[df['Predicted'] == 0]
    trues = df[df['Actual'] == df['Predicted']]
    true_positives = trues[trues['Actual'] == 1]
    true_negatives = trues.drop(true_positives.index)

    falses = df.drop(trues.index)
    false_positives_confused_display = falses[falses['Predicted'] == 1]
    false_negatives = falses.drop(false_positives_confused_display.index)

    mat = [[len(df), len(predicted_trues), len(predicted_falses)],
           [len(actual_trues), len(true_positives), len(false_negatives)],
           [len(actual_falses), len(false_positives_confused_display), len(true_negatives)]]

    df_cm = pd.DataFrame(mat, index=["Total", "Actual True", "Actual False"],
                         columns=["Total", "Predicted True", "Predicted False"])
    plt.figure(figsize=(10, 7))
    plt.ticklabel_format(useOffset=False)
    plt.title(title, fontsize=20)
    sn.set(font_scale=1.4)  # for label size

    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g')  # font size
    plt.savefig(outputDir + "confusion_matrix.png")
    # plt.show()


def load_data(filename):
    df = pd.read_csv(filename, header=0)
    return df


# %%
def disp_conf_mat_v2(tp, fp, tn, fn, title, outputDir="./runs"):
    mat = [[tp + fp + tn + fn, tp + fp, tn + fn],
           [tp + fp, tp, fn],
           [tn + fn, fp, tn]]
    print(mat)
    df_cm = pd.DataFrame(mat, index=["Total", "Actual True", "Actual False"],
                         columns=["Total", "Predicted True", "Predicted False"])
    plt.figure(figsize=(10, 7))
    plt.ticklabel_format(useOffset=False)
    plt.title(title, fontsize=20)
    sn.set(font_scale=1.4)  # for label size
    # set title font size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', vmin=(min([tp, fp, tn, fn])),
               vmax=(max([tp, fp, tn, fn])))  # font size
    # save the figure
    plt.savefig(outputDir)
    plt.show()


def calcAccuracyFromNpArr(npArr):
    print(npArr)
    correct = [x for x in npArr.tolist() if x[0] == x[1]]
    print("Correct: ", correct)


def random_select(data, target, num):
    # make a copy of data
    data = data.copy()
    # add array target as a column on the right to a copy of data
    data = np.c_[data, target]
    # shuffle the rows of the data
    np.random.shuffle(data)
    # select the first num rows of the data
    data = data[:num]
    # split the data into x and y
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def minimal_test_accuracy(model, data_x, target_out_y, completed_iterations, parameters=None, layers_size=None,
                          test_set=False, confused_needed=True):
    try:
        predictions = model.predict(data_x, target_out_y)
    except AttributeError:
        predictions = param.predict_cheaty(data_x, target_out_y, parameters, layers_size)
    # x_subset, y_subset = random_select(data_x, target_out_y, 100) if not test_set else (data_x, target_out_y)
    # x_subset, y_subset = (data_x, target_out_y)
    tp, tn, fp, fn = calcConfusionMatNums(predictions)
    correct = [x for x in predictions.to_numpy().tolist() if x[0] == x[1]]

    if test_set:
        model.test_accuracy_df.loc[len(model.test_accuracy_df)] = \
            [completed_iterations, len(correct) / len(predictions), tp, tn, fp, fn]
    else:
        model.train_accuracy_df.loc[len(model.train_accuracy_df)] = \
            [completed_iterations, len(correct) / len(predictions), tp, tn, fp, fn]
    return len(correct) / len(predictions)


def test_accuracy(model, data_x, target_out_y, completed_iterations, parameters=None, layers_size=None, test_set=False,
                  confused_needed=True):
    try:
        predictions = model.predict(data_x, target_out_y)
    except AttributeError:
        predictions = param.predict_cheaty(data_x, target_out_y, parameters, layers_size)
    # x_subset, y_subset = random_select(data_x, target_out_y, 100) if not test_set else (data_x, target_out_y)
    # x_subset, y_subset = (data_x, target_out_y)
    tp, tn, fp, fn = calcConfusionMatNums(predictions)
    correct = [x for x in predictions.to_numpy().tolist() if x[0] == x[1]]

    if test_set:
        model.test_accuracy_df.loc[len(model.test_accuracy_df)] = \
            [completed_iterations, len(correct) / len(predictions), tp, tn, fp, fn]
    else:
        model.train_accuracy_df.loc[len(model.train_accuracy_df)] = \
            [completed_iterations, len(correct) / len(predictions), tp, tn, fp, fn]
    return len(correct) / len(predictions)


def plot_accuracy(accuracy_df, num_runs, out_path):
    plt.plot(accuracy_df['Iteration'], accuracy_df['Accuracy'], label="Test")
    plt.title('Accuracy over ' + str(num_runs) + ' runs')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out_path + "accuracy.png")
    # plt.show()


def plot_test_and_training_accuracy_old(test_accuracy_df, train_accuracy_df, num_runs, out_path):
    plt.figure(num='test')
    plt.plot(test_accuracy_df['Iteration'], test_accuracy_df['Accuracy'], label="Test")
    plt.plot(train_accuracy_df['Iteration'], train_accuracy_df['Accuracy'], label="Train")
    # make the x axis logarithmic
    plt.title('Accuracy over ' + str(num_runs) + ' runs')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.savefig(out_path + "train_test_accuracy.png", dpi=1000)
    # plt.show()


def load_avg_df_to_conf_mat(df, outDir, plotName):
    # %%
    lastRow = df.tail(1)
    tp = lastRow['True Positives'].values[0]
    tn = lastRow['True Negatives'].values[0]
    fp = lastRow['False Positives'].values[0]
    fn = lastRow['False Negatives'].values[0]
    print("True Positives: {0}\tFalse Positives: {1}\tTrue Negatives: {2}\tFalse Negatives: {3}".format(tp, fp, tn, fn))
    disp_conf_mat_v2(tp, fp, tn, fn, plotName, outputDir=outDir)
    # print(lastRow)


def split_and_recombine(dataframe_array):
    print(len(dataframe_array))
    column_list = dataframe_array[0].columns.tolist()
    combined_df = pd.concat(dataframe_array, axis=1)
    result = combined_df.groupby(combined_df.columns, axis=1).sum()
    # divide all values by the number of dataframes
    result = result / len(dataframe_array)
    result.reindex(columns=column_list)
    return result


def smooth_data(dataframe):
    # numpy.convolve for data smoothing
    #     extract the "accuracy" column from the dataframe
    smoothed = np.convolve(dataframe['Accuracy'], np.ones((5,)) / 5, mode='valid')
    print(smoothed)

    # replace the accuracy column with the smoothed data
    # dataframe['Accuracy'] = smoothed
    return dataframe


def plot_test_and_training_accuracy(test_accuracy_df, train_accuracy_df, num_runs, out_path):
    plt.figure(num='test')

    # smooth the data
    # test_accuracy_df = smooth_data(test_accuracy_df)
    # array thing contains all but the last four elements of column "Accuracy"
    # array thing = test_accuracy_df['Iteration'].to_numpy()[:-4]

    # plt.plot(test_accuracy_df['Iteration'], test_accuracy_df['Accuracy'], label="Test")
    plt.plot(test_accuracy_df['Iteration'].to_numpy(),
             test_accuracy_df['Accuracy'], label="Test")
    plt.plot(train_accuracy_df['Iteration'].to_numpy(), train_accuracy_df['Accuracy'], label="Test")
    # plt.plot(train_accuracy_df['Iteration'], train_accuracy_df['Accuracy'], label="Train")
    # make the x axis logarithmic
    plt.title('Accuracy over ' + str(num_runs) + ' runs')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xscale('symlog', base=2)
    plt.legend()
    plt.savefig(out_path + "train_test_accuracy.png", dpi=1000)
    # plt.show()


def test_final_nn_accuracy(model, data_x, target_out_y):
    predictions = model.predict_cheaty(data_x, target_out_y)

    tp, tn, fp, fn = calcConfusionMatNums(predictions)
    correct = [x for x in predictions.to_numpy().tolist() if x[0] == x[1]]
    model.test_accuracy_df.loc[len(model.test_accuracy_df)] = \
        [len(correct) / len(predictions), tp, tn, fp, fn]


def gen_rand_gauss_vec(sigma, vec_x, vec_y):
    # generate random gaussian vector of size lambda_children x num_layers
    return np.random.normal(loc=0, scale=sigma, size=(vec_x, vec_y))


def false_positives(predictions):
    return predictions[(predictions['Actual'] == 0) & (predictions['Predicted'] == 1)]


def i_forgot_to_fix_the_graph():
    # %%
    train_df = pd.read_csv("../megaRuns/average_training_accuracy.csv")
    test_df = pd.read_csv("../megaRuns/average_testing_accuracy.csv")
    old_df = pd.read_csv("../data/hw_00_average_testing_accuracy.csv")
    plt.figure(num='test')
    old_df = old_df[old_df['Iteration'] < len(train_df['Iteration'].to_numpy())]
    # smooth the data
    # test_accuracy_df = smooth_data(test_accuracy_df)
    # array thing contains all but the last four elements of column "Accuracy"
    # array thing = test_accuracy_df['Iteration'].to_numpy()[:-4]
    len(test_df['Iteration'].to_numpy()) - len(old_df['Iteration'].to_numpy())
    # plt.plot(test_accuracy_df['Iteration'], test_accuracy_df['Accuracy'], label="Test")
    plt.plot(test_df['Iteration'].to_numpy()[:-9], np.convolve(test_df['Accuracy'], np.ones((10,)) / 10, mode='valid'),
             label="Test")
    plt.plot(train_df['Iteration'].to_numpy()[:-9],
             np.convolve(train_df['Accuracy'], np.ones((10,)) / 10, mode='valid'), label="Training")
    plt.plot(old_df['Iteration'].to_numpy()[:-9], np.convolve(old_df['Accuracy'], np.ones((10,)) / 10, mode='valid'),
             label="HW_00")
    # plt.plot(train_accuracy_df['Iteration'], train_accuracy_df['Accuracy'], label="Train")
    # make the x axis logarithmic
    plt.title('Accuracy over ' + str(10) + ' runs')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.savefig("./runs/Fixed_train_test_accuracy.png", dpi=500)
    # plt.show()


def make_good_graph(test, train, num_runs, out_path):
    # testing_average_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv", header=0)
    # training_average_df = pd.read_csv("./megaRuns/average_training_accuracy.csv", header=0)
    train_df = train
    test_df = test
    # %%
    # num_runs = 10
    # out_path = "./runs/"
    # test_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv")
    # train_df = pd.read_csv("./megaRuns/average_training_accuracy.csv")
    old_df = pd.read_csv("../data/hw_00_average_training_accuracy.csv")
    plt.figure(num='test')
    plt.title('Accuracy over ' + str(num_runs) + ' runs')

    len(test_df['Iteration'].to_numpy()) - len(old_df['Iteration'].to_numpy())
    plt.plot(test_df['Iteration'], test_df['Accuracy'], label="HW_01_Test")
    plt.plot(train_df['Iteration'], train_df['Accuracy'], label="HW_01_Train")
    plt.plot(old_df['Iteration'], old_df['Accuracy'], label="HW_00_Test")

    # make the x axis logarithmic
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xlim(1, (max(test_df['Iteration'].to_numpy()) - 9))
    # plt.ylim(0.48, 0.56)
    print(max(test_df['Iteration'].to_numpy()))
    plt.xscale('asinh')
    plt.legend()
    plt.savefig(out_path, dpi=1000)
    plt.show()


def plot_graph_v3(test, train, num_runs, out_path):
    train_df = train
    test_df = test
    plt.figure(num='test')
    plt.title('Accuracy over ' + str(num_runs) + ' runs')
    len(test_df['Iteration'].to_numpy())
    # plt.plot(test_accuracy_df['Iteration'], test_accuracy_df['Accuracy'], label="Test")
    plt.plot(test_df['Iteration'].to_numpy(), test_df['Accuracy'], label="Test")
    plt.plot(train_df['Iteration'].to_numpy(), train_df['Accuracy'], label="Training")

    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xlim(1, (max(test_df['Iteration'].to_numpy()) - 9))
    # plt.ylim(min
    print(max(test_df['Iteration'].to_numpy()))
    plt.xscale('asinh')
    plt.legend()
    plt.savefig(out_path, dpi=1000)
    plt.show()


def plot_graph_v4(test, train, num_runs, out_path):
    # testing_average_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv", header=0)
    # training_average_df = pd.read_csv("./megaRuns/average_training_accuracy.csv", header=0)
    train_df = train
    test_df = test
    # %%
    # num_runs = 10
    # out_path = "./runs/"
    # test_df = pd.read_csv("./megaRuns/average_testing_accuracy.csv")
    # train_df = pd.read_csv("./megaRuns/average_training_accuracy.csv")
    # hw_df = pd.read_csv("")
    hw0_df = pd.read_csv("./data/hw_00_average_training_accuracy.csv")
    hw1_df = pd.read_csv("./data/hw_01_average_testing_accuracy.csv")

    plt.figure(num='test')
    plt.title('Accuracy over ' + str(num_runs) + ' runs')

    len(test_df['Iteration'].to_numpy()) - len(hw0_df['Iteration'].to_numpy())
    plt.plot(test_df['Iteration'], test_df['Accuracy'], label="HW_02_Test")
    plt.plot(train_df['Iteration'], train_df['Accuracy'], label="HW_02_Train")
    plt.plot(hw0_df['Iteration'], hw0_df['Accuracy'], label="HW_00_Test")
    plt.plot(hw1_df['Iteration'], hw1_df['Accuracy'], label="HW_01_Test")

    # make the x axis logarithmic
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xlim(1, (max(test_df['Iteration'].to_numpy())))
    # plt.ylim(0.48, 0.56)
    print(max(test_df['Iteration'].to_numpy()))
    # logaritmic scale
    plt.xscale('asinh')
    plt.legend()
    plt.savefig(out_path, dpi=1000)
    plt.show()

def plot_graph_comparison(test, train, naive, num_runs, out_path):
    train_df = train
    test_df = test
    naive_df = naive
    plt.figure(num='test')
    plt.title('Sparse_Method_Comparison over ' + str(num_runs) + ' runs')
    plt.plot(test_df['Iteration'], test_df['Accuracy'], label="Genetic_Test")
    plt.plot(train_df['Iteration'], train_df['Accuracy'], label="Genetic_Train")
    plt.plot(naive_df['Iteration'], naive_df['Accuracy'], label="Naive_Test")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xlim(1, (max(test_df['Iteration'].to_numpy())))
    plt.xscale('asinh')
    plt.legend()
    plt.savefig(out_path, dpi=1000)
    plt.show()

def plot_graph_comparison2(test, train, num_runs, out_path):
    train_df = train
    test_df = test
    plt.figure(num='test')
    plt.title('Sparse_Method_Comparison over ' + str(num_runs) + ' runs')
    plt.plot(test_df['Iteration'], test_df['Accuracy'], label="Genetic_Test")
    plt.plot(train_df['Iteration'], train_df['Accuracy'], label="Genetic_Train")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xlim(1, (max(test_df['Iteration'].to_numpy())))
    plt.xscale('asinh')
    plt.legend()
    plt.savefig(out_path, dpi=1000)
    plt.show()


def make_conf_mat(test):
    test_tp, test_tn, test_fp, test_fn, test_positives, test_negatives = test.tail(1)['True Positives'].values[0], \
    test.tail(1)['True Negatives'].values[0], test.tail(1)['False Positives'].values[0], \
    test.tail(1)['False Negatives'].values[0], test.tail(1)['Positives'].values[0], test.tail(1)['Negatives'].values[0]

    test_mat = [[test_positives + test_negatives, test_positives, test_negatives],
                [test_positives, test_tp, test_fp],
                [test_negatives, test_fn, test_tn]]
    return test_mat

def plot_conf_comparison(test, train, naive, out_path):
    test_conf_mat = make_conf_mat(test)
    test_df_conf_mat = pd.DataFrame(test_conf_mat, index=["Total", "Actual True", "Actual False"],
                         columns=["Total", "Predicted True", "Predicted False"])
    train_conf_mat = make_conf_mat(train)
    train_df_conf_mat = pd.DataFrame(train_conf_mat, index=["Total", "Actual True", "Actual False"],
                            columns=["Total", "Predicted True", "Predicted False"])

    naive_conf_mat = make_conf_mat(naive)
    naive_df_conf_mat = pd.DataFrame(naive_conf_mat, index=["Total", "Actual True", "Actual False"],
                            columns=["Total", "Predicted True", "Predicted False"])


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Confusion Matrix Comparison', fontsize=20)


    # set figsize of all subplots to 10, 7
    fig.set_figheight(7)
    fig.set_figwidth(24)
    ax1.set_title("Genetic_Train")
    ax2.set_title("Genetic_Test")
    ax3.set_title("Naive_Approach")
    sn.heatmap(test_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax1)
    sn.heatmap(train_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax2)
    sn.heatmap(naive_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax3)
    # ax1.plot(sn.heatmap(test_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax1))
    # ax2.plot(sn.heatmap(train_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax2))
    # ax3.plot(sn.heatmap(naive_df_conf_mat, annot=True, annot_kws={"size": 16}, fmt='g', ax=ax3))
    plt.savefig(out_path, dpi=1000)
    plt.show()
