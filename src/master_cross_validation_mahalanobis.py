import sys
import time
import csv

from src.data import save_results, read_data
from src.instance_selection_methods import mahalanobis
from src.models import train_model, prediction
from sklearn import decomposition

# metric, is_kernel, dataset, ?categorization_mode
print('Argument List:', str(sys.argv[1:]))
dataset_name = sys.argv[3]
score_metric = 1 if dataset_name in ["Abalone"] else 2 if dataset_name in ['Ionosphere', 'Cancer', 'Mammographic'] else 0
categorization_mode = None
if len(sys.argv[1:]) == 4:
    categorization_mode = sys.argv[-1]
start_time = time.perf_counter()
dataset = read_data.get_dataset(dataset_name) if categorization_mode is None \
    else read_data.get_dataset(dataset_name, categorization_mode)
if "Iris" in dataset_name:
    pca = decomposition.PCA()
    pca.n_components = 2
    dataset.data = pca.fit_transform(dataset.data).tolist()
categorization_mode = "" if categorization_mode is None else categorization_mode
dataset_name = dataset_name + categorization_mode
if len(dataset) > 0:
    print("Got data for {} in {}s".format(dataset_name, time.perf_counter() - start_time))
    is_kernel = int(sys.argv[2]) == 1
    if is_kernel:
        print("Not implemented kernel distance yet")
    else:
        metric = sys.argv[1]
        start_time = time.perf_counter()
        validation_sets, p_values = read_data.get_cross_validation_sets(dataset_name)
        if len(validation_sets) > 0:
            print("Got validation sets in {}".format(time.perf_counter() - start_time))
            validation = None
            total_time = time.perf_counter()
            selection_rows =[]
            for validation in validation_sets:
                start_time = time.perf_counter()
                set_id = validation['id']
                index_per_label = [[int(i) for i in nested_array] for nested_array in validation['potential']]
                datas = [[[dataset.data[i] for i in nested_array] for nested_array in index_per_label],
                         [dataset.data[int(i)] for i in validation['test']]]
                unique_labels = [int(label) for label in validation['labels']] if validation['labels'][0].isdigit() \
                    else validation['labels']
                labels = [unique_labels, [dataset.target[int(i)] for i in validation['test']]]
                print("prepared validation set in {}s".format(time.perf_counter() - start_time))
                validation_size = 0.20
                k_values = [3, 5, 10]

                for p_key in p_values.keys():
                    mahalanobis_scores_knn = []
                    selection_seeds_knn = []
                    p_value = (p_key, p_values[p_key])
                    p_per_class = []
                    for x_data, label in zip(datas[0], labels[0]):
                        p = len(x_data) * p_value[0]
                        difference = p - int(p)
                        p = int(p) if difference < 0.5 else int(p) + 1
                        p_per_class.append(p)
                    missing = int(p_value[1] - sum(p_per_class))
                    if missing > 0:
                        for i in range(abs(missing)):
                            p_per_class[-(i + 1)] += 1
                    elif missing < 0:
                        for i in range(abs(missing)):
                            p_per_class[-(i + 1)] -= 1
                    if sum(p_per_class) != p_value[1]:
                        print("Error in p_per_class for {}".format(p_value))
                        continue
                    start_time = time.perf_counter()
                    mahalanobis_set, mahalanobis_label, removed_idx = mahalanobis.mahalanobis_outliers(
                        datas[0], labels[0], p_per_class, 0, inverted=False)
                    selection_rows.append([set_id, p_value, removed_idx, unique_labels])
                    for k in k_values:
                        mahalanobis_classifier = train_model.train_knn(mahalanobis_set, mahalanobis_label, k)
                        score = prediction.predict_scores(datas[1], labels[1], mahalanobis_classifier, score_metric)
                        selection_seeds_knn.append(0)
                        mahalanobis_scores_knn.append(score)

                    n_results_knn = len(mahalanobis_scores_knn)
                    empty_result = [([0, 0], 0.0, 0.0)]
                    result_knn = [([labels[1]] * n_results_knn, empty_result * n_results_knn, empty_result * n_results_knn,
                                   mahalanobis_scores_knn, [set_id] * n_results_knn, selection_seeds_knn,
                                   [validation_size] * n_results_knn, [p_value] * n_results_knn, k_values)]
                    print("finished calculating results in {}".format(time.perf_counter() - start_time))
                    save_results.save_cross_validation_results("{}_{}_knn_mahalanobis".format(dataset_name, metric),
                                                               result_knn)
                    print("Saved {} results for knn:".format(n_results_knn),
                          str([set_id, dataset_name, metric, is_kernel, p_value, k_values]))
            print("Got and saved all results in {}".format(time.perf_counter() - total_time))
            filename = "data/results/{}_{}_mahalanobis_selection.csv".format(dataset_name, metric)
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['split seed', 'p', 'selection', 'labels'])
                writer.writerows(selection_rows)
        else:
            print("Have to create validation sets first for {}.".format(dataset_name))
else:
    print("Couldn't get the data from the dataset {}".format(dataset_name))
