from sklearn.model_selection import KFold
from src.data.save_results import save_cross_validation_sets, save_p_values
import numpy as np
import math


def create_split(dataset_name, data, n_split=5, p_values=None, test_seed=221):
    p_values = [1/2, 1/4, 1/8] if p_values is None else p_values
    data_index = list(range(len(data.data)))

    data_left = []
    labels_order = []
    tests = []
    test_seeds = []
    train_size = 0

    for i in range(10):
        kf = KFold(n_splits=n_split, shuffle=True, random_state=test_seed+i)
        sets = kf.split(data_index)
        for train, test in sets:
            train_labels = [data.target[i] for i in train]
            labels = np.unique(train_labels)
            index_per_label = [[train[i] for i, value in enumerate(train_labels) if value == label] for label in labels]
            sorted_order = np.argsort([len(data) for data in index_per_label])
            index_per_label = [index_per_label[i] for i in sorted_order]
            labels = [labels[i] for i in sorted_order]

            labels_order.append(labels)
            data_left.append(index_per_label)
            tests.append(test.tolist())
            test_seeds.append(test_seed + i)
            train_size = len(train)

    save_cross_validation_sets(dataset_name, labels_order, data_left, tests, list(range(len(tests))), test_seeds)

    p_values = [(p, int(math.ceil(train_size * p))) for p in p_values]
    save_p_values(dataset_name, p_values)
