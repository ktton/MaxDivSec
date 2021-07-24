import sys
import time
from datetime import datetime, timedelta
import numpy as np

from src.data import read_data
from src.instance_selection_methods.mahalanobis import calculate_inv_cov_matrix

from sklearn.metrics import pairwise_distances
from sklearn import decomposition

# metric, is_kernel, dataset, ?categorization_mode
print('Argument List:', str(sys.argv[1:]))
dataset_name = sys.argv[3]
categorization_mode = None
if len(sys.argv[1:]) == 4:
    categorization_mode = sys.argv[-1]
start_time = time.perf_counter()
dataset = read_data.get_dataset(dataset_name) if categorization_mode is None \
    else read_data.get_dataset(dataset_name, categorization_mode)
if "Iris" in dataset_name:
    pca = decomposition.PCA()
    pca.n_components = 2
    dataset.data = pca.fit_transform(dataset.data)
categorization_mode = "" if categorization_mode is None else categorization_mode
score_metric = 1 if "Abalone" in dataset_name else 2 if "Adult" in dataset_name else 0
if len(dataset) > 0:
    print("Got data for {} in {}s".format(dataset_name, time.perf_counter() - start_time))
    is_kernel = int(sys.argv[2]) == 1
    if is_kernel:
        print("Not implemented kernel distance yet")
    else:
        metric = sys.argv[1]
        start_time = time.perf_counter()
        validation_sets, p_values = read_data.get_cross_validation_sets(dataset_name + categorization_mode)
        improvements = []
        if len(validation_sets) > 0:
            print("Got validation sets in {}".format(time.perf_counter() - start_time))
            validation = None
            total_time = time.perf_counter()

            script_filename = "OBMA/run_OBMA.sh"
            with open(script_filename, 'w') as file:
                file.write("#!/bin/bash\n")

            total_script_time = 0
            for validation in validation_sets:
                start_time = time.perf_counter()
                set_id = validation['id']
                index_per_label = [[int(i) for i in nested_array] for nested_array in validation['potential']]
                datas = [[dataset.data[i].tolist() for i in nested_array] for nested_array in index_per_label]
                unique_labels = [int(label) for label in validation['labels']] if validation['labels'][0].isdigit() \
                    else validation['labels']
                print("prepared validation set {} in {}s".format(set_id, time.perf_counter() - start_time))

                split_distance = []
                distance_rows_class = []
                start_time = time.perf_counter()
                for label in range(len(unique_labels)):
                    training = datas[label]
                    if len(training) > 1:
                        if metric == "mahalanobis":
                            if len(training) > 2:
                                inv_cov = calculate_inv_cov_matrix(training)
                                split_distance.append(pairwise_distances(training, metric=metric, VI=inv_cov))
                            else:
                                split_distance.append(np.array([[0., 1.], [1., 0.]]))
                        else:
                            split_distance.append(pairwise_distances(training, metric=metric))
                    else:
                        split_distance.append(np.array([[0.]]))
                    distance_rows = []
                    for i in range(len(training)):
                        for j in range(i + 1, len(training)):
                            distance_rows.append("{} {} {}\n".format(i, j, split_distance[-1][i][j]))
                    distance_rows_class.append(distance_rows)
                print("Got distance matrix in {}".format(time.perf_counter() - start_time))
                biggest_p = max(p_values.keys())

                for p_value in p_values.items():
                    p_per_class = []
                    size_per_class = []
                    for i, (x_data, label) in enumerate(zip(datas, unique_labels)):
                        p = len(x_data) * p_value[0]
                        difference = p - int(p)
                        p = int(p) if difference < 0.5 else int(p) + 1
                        p_per_class.append(p)
                        size_per_class.append(len(x_data))
                    if p_value[0] == biggest_p:
                        for i in range(len(size_per_class)):
                            while p_per_class[i] / size_per_class[i] > 0.5:
                                p_per_class[i] -= 1
                    else:
                        missing = int(p_value[1] - sum(p_per_class))
                        if missing > 0:
                            for i in range(abs(missing)):
                                p_per_class[-(i + 1)] += 1
                        elif missing < 0:
                            for i in range(abs(missing)):
                                p_per_class[-(i + 1)] -= 1
                    for label, distance_rows, size, p in zip(unique_labels, distance_rows_class, size_per_class,
                                                             p_per_class):
                        if p < 2:
                            continue
                        filename = "OBMA/instances/MDG-c/{0}_{1}_L{2}_p{3}.txt".format(dataset_name, set_id, label,
                                                                                       str(p_value[0]).replace(".", ""))
                        with open(filename, 'w') as f:
                            f.write("{} {}\n".format(size, p))
                            f.writelines(distance_rows)
                        time_limit = 30 if size < 50 \
                            else 60 if size < 150 \
                            else 120 if size < 300 \
                            else 180 if size < 500 \
                            else 300 if size < 600 \
                            else 420 if size < 700 \
                            else 540 if size < 800 \
                            else 600  # 1k+
                        total_script_time += (time_limit * 2)
                        with open(script_filename, 'a') as file:
                            file.write("date\n")
                            file.write("./obma {0}_{1}_L{2}_p{3}.txt MDG-c {4} 2\n".format(dataset_name, set_id, label,
                                                                                           str(p_value[0]).replace(".",
                                                                                                                   ""),
                                                                                           time_limit))
            with open(script_filename, 'a') as file:
                file.write("echo $'\\a'\n")
            hours_to_run = total_script_time / 60 / 60
            print("{}h".format(hours_to_run))
            current_date_and_time = datetime.now()
            hours_added = timedelta(hours=hours_to_run)
            future_date_and_time = current_date_and_time + hours_added
            print(future_date_and_time)
