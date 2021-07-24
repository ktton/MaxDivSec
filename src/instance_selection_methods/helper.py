import numpy as np


def get_p_dispersion_pair(distance_matrix, already_chosen=None):
    if already_chosen is None:
        already_chosen = []
    new_pair = []
    max_value = -np.inf
    for n_row, row in enumerate(distance_matrix):
        skip_row = n_row in already_chosen
        if skip_row:
            continue
        else:
            index = np.argmin(row)
            min_value = row[index]
            if max_value < min_value:
                max_value = min_value
                new_pair = [n_row, index] if len(already_chosen) == 0 else [n_row, already_chosen[index]]
    return new_pair


def split_class(x_data, y_data, distance_matrix=None):
    labels = np.unique(y_data)
    index_per_label = [[i for i, value in enumerate(y_data) if value == label] for label in labels]

    split_data = [[x_data[i] for i in index_per_label[j]] for j in range(len(labels))]
    sorted_order = np.argsort([len(data) for data in split_data])
    split_data = [split_data[i] for i in sorted_order]

    labels = [labels[i] for i in sorted_order]

    split_distance = [[[0 if i == j else distance_matrix[i][j-i-1] if i < j else distance_matrix[j][i-j-1]
                        for j in index_per_label[label]] for i in index_per_label[label]] if distance_matrix is not None
                      else None for label in range(len(labels))]

    split_distance = [split_distance[i] for i in sorted_order]
    return split_data, labels, split_distance, index_per_label
