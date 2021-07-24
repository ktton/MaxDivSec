import sys
import time
import numpy as np
import csv
import os

from sklearn import decomposition
from sklearn.metrics import pairwise_distances

from src.data import save_results, read_data
from src.instance_selection_methods.mahalanobis import calculate_inv_cov_matrix
from src.models import train_model, prediction

# metric, is_kernel, fake_instance, best_selection, dataset, ?categorization_mode
print('Argument List:', str(sys.argv[1:]))
dataset_name = sys.argv[5]
categorization_mode = None
if len(sys.argv[1:]) == 6:
    categorization_mode = sys.argv[-1]
start_time = time.perf_counter()
dataset = read_data.get_dataset(dataset_name) if categorization_mode is None \
    else read_data.get_dataset(dataset_name, categorization_mode)
if "Iris" in dataset_name:
    pca = decomposition.PCA()
    pca.n_components = 2
    dataset.data = pca.fit_transform(dataset.data)
is_abalone = "Abalone" in dataset_name
is_adult = "Adult" in dataset_name

categorization_mode = "" if categorization_mode is None else categorization_mode
dataset_name_cat = dataset_name + categorization_mode
score_metric = 1 if dataset_name in ["Abalone"] else 2 if dataset_name in ['Ionosphere', 'Cancer', 'Mammographic'] else 0
if len(dataset) > 0:
    print("Got data for {} in {}s".format(dataset_name_cat, time.perf_counter() - start_time))
    is_kernel = int(sys.argv[2]) == 1
    if is_kernel:
        print("Not implemented kernel distance yet")
    else:
        metric = sys.argv[1]
        fake_instance = sys.argv[3]
        best_selection = sys.argv[4]
        start_time = time.perf_counter()
        validation_sets, p_values = read_data.get_cross_validation_sets(dataset_name_cat)
        if len(validation_sets) > 0:
            print("Got validation sets in {}".format(time.perf_counter() - start_time))
            error = False
            total_time = time.perf_counter()
            for validation in validation_sets:
                improvement_rows = []
                start_time = time.perf_counter()
                set_id = validation['id']
                index_per_label = [[int(i) for i in nested_array] for nested_array in validation['potential']]
                datas = [[dataset.data[i].tolist() for i in nested_array] for nested_array in index_per_label]
                test_data = [dataset.data[int(i)].tolist() for i in validation['test']]
                test_labels = [dataset.target[int(i)] for i in validation['test']]
                unique_labels = [int(label) for label in validation['labels']] if validation['labels'][0].isdigit() \
                    else validation['labels']
                print("prepared validation set {} in {}s".format(set_id, time.perf_counter() - start_time))
                biggest_p = max(p_values.keys())
                for p_value in p_values.items():
                    if p_value[0] == 0.75:
                        continue
                    training_data = []
                    training_labels = []
                    p_per_class = []
                    selections = []
                    size_per_class = []
                    for x_data in datas:
                        size_per_class.append(len(x_data))
                        p = len(x_data) * p_value[0]
                        difference = p - int(p)
                        p = int(p) if difference < 0.5 else int(p) + 1
                        p_per_class.append(p)
                    if p_value[0] == biggest_p:
                        for idx in range(len(p_per_class)):
                            while p_per_class[idx] / size_per_class[idx] > 0.5:
                                p_per_class[idx] -= 1
                    else:
                        missing = int(p_value[1] - sum(p_per_class))
                        if missing > 0:
                            for i in range(abs(missing)):
                                p_per_class[-(i + 1)] += 1
                        elif missing < 0:
                            for i in range(abs(missing)):
                                p_per_class[-(i + 1)] -= 1
                    for i, (label, data, p) in enumerate(zip(unique_labels, datas, p_per_class)):
                        label_str = 1 if is_adult and '>' in label else 0 if is_adult else label
                        dataset_name_str = dataset_name + "_"+metric if is_adult else dataset_name
                        filename = "OBMA/results/{0}/{5}_{1}_L{2}_p{3}.txt.res{4}".format(fake_instance, set_id, label_str,
                                                                                          str(p_value[0]).replace('.', ''),
                                                                                          best_selection, dataset_name_str)
                        if not os.path.exists(filename):
                            print("Missing file {} with p={}".format(filename, p))
                            if p == 0:
                                training_data += data
                                training_labels += [label] * len(data)
                                selections.append([])
                            elif p == 1:
                                if metric == "mahalanobis":
                                    if len(data) > 2:
                                        inv_cov = calculate_inv_cov_matrix(data)
                                        distance = pairwise_distances(data, metric=metric, VI=inv_cov)
                                    else:
                                        distance = np.array([[0., 1.], [1., 0.]])
                                else:
                                    distance = pairwise_distances(data, metric=metric)
                                max_per_row = np.amax(distance, axis=1)
                                first = np.argmax(max_per_row)
                                selections.append([first])
                                training = [data[idx] for idx in range(len(data)) if idx != first]
                                training_data += training
                                training_labels += [label] * len(training)
                            else:
                                print("Shouldn't be missing files for OBMA".format(set_id, p_value))
                                error = True
                                break
                        else:
                            with open(filename, 'r') as file:
                                selection = [int(idx) for idx in file.readlines()[-1].split(' ')[:-1]]
                                selections.append(selection)
                                training = [data[idx] for idx in range(len(data)) if idx not in selection]
                                training_data += training
                                training_labels += [label] * len(training)
                    if error:
                        error = False
                        continue
                    scores_knn = []
                    selection_seeds_knn = []
                    k_values = [3, 5, 10]
                    validation_size = 0.20
                    for k in k_values:
                        classifier = train_model.train_knn(training_data, training_labels, k)
                        score = prediction.predict_scores(test_data, test_labels, classifier, score_metric)
                        scores_knn.append(score)
                        selection_seeds_knn.append(0)
                    n_results_knn = len(scores_knn)
                    empty_result = [([0, 0], 0.0, 0.0)]

                    result_knn = [([test_labels] * n_results_knn, empty_result * n_results_knn, scores_knn,
                                   empty_result * n_results_knn, [set_id] * n_results_knn, selection_seeds_knn,
                                   [validation_size] * n_results_knn, [p_value] * n_results_knn, k_values)]

                    print(
                        "finished calculating results in {}".format(time.perf_counter() - start_time))
                    filename = "{}_euclidean_{}_knn_max_diversity_OBMA".format(dataset_name_cat, metric) \
                        if metric == "mahalanobis" else "{}_{}_knn_max_diversity_OBMA".format(dataset_name_cat, metric)
                    save_results.save_cross_validation_results(filename, result_knn)
                    print("Saved {} results for knn:".format(n_results_knn),
                          str([set_id, dataset_name_cat, metric, is_kernel, p_value, k_values]))
                    improvement_rows.append(['OBMA', set_id, p_value, selections.copy(), unique_labels.copy(), [np.NaN]])
                filename_improvement = "data/results/{}_euclidean_{}_max_diversity_improvements.csv".format(dataset_name_cat,
                                                                                                            metric) \
                    if metric == "mahalanobis" else "data/results/{}_{}_max_diversity_improvements.csv".format(
                    dataset_name_cat,
                    metric)
                exist = os.path.exists(filename_improvement)
                with open(filename_improvement, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    if not exist:
                        writer.writerow(['method', 'split seed', 'p', 'selection', 'labels', 'improvement'])
                    writer.writerows(improvement_rows)

