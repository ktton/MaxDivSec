import numpy as np
import sys


def remove_example(x_data, y_data, remove_set):
    remove_set.sort(reverse=True)
    training_set = x_data.tolist() if isinstance(x_data, np.ndarray) else x_data.copy()
    label_set = y_data.tolist() if isinstance(y_data, np.ndarray) else y_data.copy()
    for index in remove_set:
        del training_set[index]
        del label_set[index]

    return training_set, label_set


def mahalanobis_outliers(x_data, labels, p_per_class, selection_seed, index_only=False, inverted=True):

    training_set = []
    label_set = []
    new_p = [len(data) - p for p, data in zip(p_per_class, x_data)] if inverted else p_per_class
    idx_to_remove = []
    for data, label, p in zip(x_data, labels, new_p):
        if p == len(data):
            idx_to_remove.append(list(range(p)))
            continue
        if p == 0:
            idx_to_remove.append([])
            if not index_only:
                sub_training_set = np.copy(data).tolist()
                sub_labels = [label] * len(sub_training_set)
        elif 0 < p < len(data):
            inv_covariance_matrix = calculate_inv_cov_matrix(data)
            vars_mean = []
            for i in range(len(data)):
                vars_mean.append(list(np.mean(data, axis=0)))
            diff = np.array(data) - vars_mean
            distance = []
            for i in range(len(diff)):
                distance.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

            sorted_distance = distance.copy()
            sorted_distance.sort()
            to_remove = sorted_distance[-p:]
            to_remove = [distance.index(outlier) for outlier in to_remove]

            idx_to_remove.append(to_remove)
            if not index_only:
                sub_training_set, sub_labels = remove_example(data, [label] * len(data), to_remove)
        elif p > len(data) or p < 0:
            print("Error p is incorrect: {}".format(p))
            return None, None
        if not index_only:
            training_set += sub_training_set
            label_set += sub_labels
    if index_only:
        return idx_to_remove
    return training_set, label_set, idx_to_remove


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def calculate_inv_cov_matrix(A):
    covariance_matrix = np.cov(A, rowvar=False)
    try:
        inverted_cov = np.linalg.inv(covariance_matrix)
        if is_pos_def(inverted_cov):
            return inverted_cov
    except np.linalg.LinAlgError:
        print("LinAlgError")
    print("Couldn't not invert matrix normally. Will use approximation")
    u, s, vh = np.linalg.svd(covariance_matrix, full_matrices=True)
    inv_s = np.array([1 / d if d > sys.float_info.epsilon else 0 for d in s])
    return np.matmul(np.matmul(vh.T, np.diag(inv_s)), u.T)
