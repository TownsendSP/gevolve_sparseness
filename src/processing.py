import pandas as pd
from scipy.io import arff

debug = False
def print_unique_values_of_df_column(df, col_name):
    print(df[col_name].unique())


def split_data(df):
    good = df[df['Class'] == 1]
    bad = df[df['Class'] == 0]

    train_good = good.sample(frac=0.7, random_state=1)
    test_good = good.drop(train_good.index)
    train_bad = bad.sample(frac=0.7, random_state=1)
    test_bad = bad.drop(train_bad.index)

    train = pd.concat([train_good, train_bad])
    test = pd.concat([test_good, test_bad])

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    train.to_csv('./megaRuns/runs/train.csv', index=False)
    test.to_csv('./megaRuns/runs/test.csv', index=False)

    train_y = train['Class']
    train_x = train.drop(['Class'], axis=1)
    test_y = test['Class']
    test_x = test.drop(['Class'], axis=1)

    print(str.format("Train_x_pos: {0}\tTrain_x_neg: {1}\tTest_x_pos: {2}\tTest_x_neg: {3}",
                     train_x[train_y == 1].shape[0],
                     train_x[train_y == 0].shape[0],
                     test_x[test_y == 1].shape[0],
                     test_x[test_y == 0].shape[0]
                     )) if debug else None

    return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()


def arff_to_df(filename):
    data, meta = arff.loadarff(filename)
    df = pd.DataFrame(data)
    return df

def drop_class_column(df):
    return df['Class'], df.drop(['Class'], axis=1)

def normalise_beans(bean_df):
    # classes to keep:
    # retain all rows where class = 'SEKER' or 'HOROZ'
    bean_df = bean_df[bean_df['Class'].isin([b'SEKER', b'HOROZ'])]
    print(bean_df.shape)
    seker_num = bean_df[bean_df['Class'] == b'SEKER'].shape[0]
    horoz_num = bean_df[bean_df['Class'] == b'HOROZ'].shape[0]
    print(str.format("Before Normalization:\n\tSEKER count: {0}\n\tHOROZ count: {1}",
                     bean_df[bean_df['Class'] == b'SEKER'].shape[0], bean_df[bean_df['Class'] == b'HOROZ'].shape[0]))
    bean_df = pd.concat([bean_df,
                         bean_df[bean_df['Class'] == b'HOROZ'].sample(
                             n=seker_num - horoz_num, random_state=1)])
    print(str.format("After Normalization:\n\tSEKER count: {0}\n\tHOROZ count: {1}",
                     bean_df[bean_df['Class'] == b'SEKER'].shape[0], bean_df[bean_df['Class'] == b'HOROZ'].shape[0]))
    return bean_df


def scale_replace_beans(bean_df):
    bean_df = bean_df.replace(b'SEKER', 0)
    bean_df = bean_df.replace(b'HOROZ', 1)
    for (column, columnData) in bean_df.items():
        bean_df[column] -= columnData.min()
        bean_df[column] /= (columnData.max())
    return bean_df


def recombine_by_title(df_list):
    output_list = []
    for title, df in df_list:
        for col_name in df:
            col_data = df[col_name]
            asList = col_data.tolist()
            print(asList)
            if col_name != 'Class':
                output_list.append((title, col_name, asList))
    # print(output_list)

    col_dict = {}

    for title, column_name, col_data in output_list:
        if column_name in col_dict:
            col_dict[column_name]['title'].append(title)
            col_dict[column_name]['data'].append(col_data)
        else:
            col_dict[column_name] = {'title': [title], 'data': [col_data]}
    df_out_list = []
    list_Proper_tuples = []
    for col_name in col_dict:
        curr_plot_name = col_name
        curr_plot_data = col_dict[col_name]['data']
        curr_plot_title = col_dict[col_name]['title']
        curr_tuple_list = [(curr_plot_title[i], curr_plot_data[i]) for i in range(len(curr_plot_title))]
        # for i in range(len(curr_plot_title)):
        #     curr_tuple_list.append((curr_plot_title[i], curr_plot_data[i]))
        list_Proper_tuples.append((curr_plot_name, curr_tuple_list))
    print(list_Proper_tuples[0])
    for col_name, tuple_list in list_Proper_tuples:
        curr_df = pd.DataFrame.from_dict(dict(tuple_list))
        df_out_list.append((col_name, curr_df))

        # col_data = list(zip(col_dict[col_name]['data'], col_dict[col_name]['title']))
        # df_out_list.append((col_name, pd.DataFrame(col_data, columns=['Class', col_name])))
    return df_out_list