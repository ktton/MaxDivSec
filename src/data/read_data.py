import pandas as pd
from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


def get_iris():
    return datasets.load_iris()


def get_abalone(categorization_mode=0):
    df_abalone = pd.read_csv('data/external/abalone.csv')
    df_abalone.drop('Sex', axis=1, inplace=True)
    if categorization_mode == 1:
        df_abalone['Rings'] = np.array(
            list(map(lambda x: 0 if x < 5 else 1 if x < 10 else 2 if x < 15 else 3 if x < 20 else 4,
                     df_abalone['Rings'])))
    elif categorization_mode == 2:
        df_abalone['Rings'] = np.array(
            list(map(lambda x: 0 if x < 6 else 1 if x < 14 else 2, df_abalone['Rings'])))
    abalone = Bunch()
    abalone.data = StandardScaler().fit_transform(df_abalone.drop('Rings', axis=1))
    abalone.target = np.array(df_abalone['Rings'])
    return abalone


def get_synthetic(name):
    df_synthetic = pd.read_csv('data/external/Synthetic{}.csv'.format(name))
    synthetic = Bunch()
    synthetic.target = np.array(df_synthetic['Label'])
    synthetic.data = np.array(df_synthetic.drop('Label', axis=1))
    return synthetic


def get_breast_cancer():
    df_cancer = pd.read_csv('data/external/breast-cancer-wisconsin.csv')
    df_cancer.drop('code_number', axis=1, inplace=True)
    cancer = Bunch()
    cancer.data = StandardScaler().fit_transform(df_cancer.drop('class', axis=1))
    cancer.target = np.array(df_cancer['class'])
    return cancer


def get_ionoshpehere():
    df = pd.read_csv('data/external/ionosphere.csv')
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('y', axis=1))
    dataset.target = np.array(df['y'])
    return dataset


def get_leaf():
    df = pd.read_csv('data/external/leaf.csv')
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('class', axis=1))
    dataset.target = np.array(df['class'])
    return dataset


def get_seed():
    df = pd.read_csv('data/external/seeds_dataset.csv')
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('class', axis=1))
    dataset.target = np.array(df['class'])
    return dataset


def get_mammographic():
    df = pd.read_csv('data/external/mammographic_masses.csv')
    df.replace('?', np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('class', axis=1))
    dataset.target = np.array(df['class'])
    return dataset


def get_dermatology():
    df = pd.read_csv('data/external/dermatology.csv')
    df.replace('?', np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('class', axis=1))
    dataset.target = np.array(df['class'])
    return dataset


def get_contraceptive():
    df = pd.read_csv('data/external/cmc.csv')
    df.replace('?', np.nan, inplace=True)
    df = df.dropna().reset_index(drop=True)
    dataset = Bunch()
    dataset.data = StandardScaler().fit_transform(df.drop('class', axis=1))
    dataset.target = np.array(df['class'])
    return dataset


switcher = {
    "Iris": get_iris,
    "Abalone": get_abalone,
    "Synthetic": get_synthetic,
    "Seed": get_seed,
    "Ionosphere": get_ionoshpehere,
    "Cancer": get_breast_cancer,
    "Mammographic": get_mammographic,
    "Dermatology": get_dermatology,
    "Contraceptive": get_contraceptive
}


def get_dataset(dataset, *args):
    func = switcher.get(dataset, None)
    if func is None:
        return Bunch()
    return func(*args)


def parse_score(original_df, method_name, other_list):
    df_temp = original_df.copy()
    for other in other_list:
        df_temp = df_temp.drop(other + ' score', axis=1)

    parsing = [x[2:-1].split('], ') for x in df_temp[method_name + ' score']]
    predictions = [[int(x) if x.isdigit() else x.strip("'") for x in y[0].split(', ')] for y in parsing]

    ground_truth = [[int(label) if label.isdigit() else label.strip("'") for label in row[1:-1].split(', ')]
                    for row in df_temp['Ground truth']]
    scores = [x[1].split(',') for x in parsing]
    accuracy = [float(x[0]) for x in scores]
    kappa = [float(x[1]) for x in scores]

    df_temp['Method'] = [method_name] * len(original_df.index)
    df_temp['Accuracy'] = accuracy
    df_temp['Prediction'] = predictions
    df_temp['Kappa'] = kappa
    df_temp['Ground truth'] = ground_truth

    df_temp = df_temp.drop(method_name + ' score', axis=1)

    return df_temp


def read_score_data(dataset_name, classifier_name):
    filename = "data/results/{}_{}.csv".format(dataset_name, classifier_name)
    if os.path.exists(filename):
        data = pd.read_csv(filename)

        methods = ['Random', 'p-dispersion', 'Hybrid']
        df = None

        for method in methods:
            other_methods = list(methods)
            other_methods.remove(method)
            df = pd.concat([df, parse_score(data, method, other_methods)], ignore_index=True)
    else:
        print("File {} not found".format(filename))
        data = None
        df = None
    return data, df


def parse_potential_set(potential_set_string):
    nested_array = potential_set_string[:-1].split('], ')
    potential_set = [[int(x) for x in y[1:].split(',')] for y in nested_array]
    return potential_set


def get_cross_validation_sets(dataset_name):
    filename = "data/interim/{}_cross_validation_set.csv".format(dataset_name)
    filename2 = "data/interim/{}_p_values.csv".format(dataset_name)
    if os.path.exists(filename) and os.path.exists(filename2):
        data = pd.read_csv(filename)
        validation_sets = [{'labels': [x.strip("'") for x in row[1][1:-1].split(', ')],
                            'potential': parse_potential_set(row[2][1:-1]),
                            'test': row[3][1:-1].split(', '), 'id': row[4]} for row in data[data.columns].values]
        data = pd.read_csv(filename2)
        p_values = {p: data[data['p'] == p]['Training size'].values[0] for p in data['p'].unique()}
        return validation_sets, p_values
    else:
        print("No cross-validation sets or p values found for {}".format(dataset_name))
        return [], []


def get_cross_validation_results(dataset_name):
    filename = "data/results/{}_cross_validation.csv".format(dataset_name)
    if os.path.exists(filename):
        data = pd.read_csv(filename)
        methods = ['Random', 'p-dispersion', 'Mahalanobis outlier detection']
        methods = methods + ['max-diversity'] if 'max-diversity score' in data.columns else methods
        df = None

        for method in methods:
            other_methods = list(methods)
            other_methods.remove(method)
            df = pd.concat([df, parse_score(data, method, other_methods)], ignore_index=True)

        return data, df
    print("Couldn't find the cross-validation results for {}".format(dataset_name))
    return None, None
