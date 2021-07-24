import os
import csv
import pandas as pd


def save(filename, csv_header, csv_content, overwrite=False):
    already_exist = False if overwrite else os.path.exists(filename)
    mode = "a" if already_exist else "w"
    with open(filename, mode, newline='') as f:
        writer = csv.writer(f, delimiter=',')
        if not already_exist:
            writer.writerow(csv_header)
        writer.writerows(csv_content)


def save_scores(dataset_name, results, classifier_type):
    csv_header = ["Dataset", "Random score", "p-dispersion score", "Hybrid score", "Random seed",
                  "Selection seed", "Test size", "Selection size", "Test set"]
    csv_content = []
    for (ground_truth, random_scores, p_dispersion_scores, hybrid_scores, random_seeds, selection_seed,
         training_size, selection_size) in results:
        for i in range(len(random_scores)):
            csv_content.append(
                [dataset_name, random_scores[i], p_dispersion_scores[i], hybrid_scores[i], random_seeds[i],
                 selection_seed[i], training_size, selection_size, list(ground_truth[0])])
    filename = "data/results/{}_{}.csv".format(dataset_name, classifier_type)
    save(filename, csv_header, csv_content)


def save_selection(dataset_name, metric, label_set, percentage, training_size):
    csv_header = ["Dataset", "Metric", "Label set", "Percentage", "Training size"]
    csv_content = [[dataset_name, metric, label_set, percentage, training_size]]
    filename = "data/interim/{}_training_set.csv".format(dataset_name)
    save(filename, csv_header, csv_content)


# the training set is composed of the index each chosen instance
def save_training_set(dataset_name, metric, training_set, label_set, training_size, time):
    csv_header = ["Dataset", "Metric", "Training set", "Label set", "Training size", "Time"]
    csv_content = [dataset_name, metric, training_set, label_set, training_size, time]
    filename = "data/interim/{}_p_dispersion_set.csv".format(dataset_name)
    save(filename, csv_header, csv_content)


def save_cross_validation_sets(dataset_name, labels_order, data_left, tests, set_id, test_seeds):
    csv_header = ["Dataset", "Labels", "Potential training set", "Test set", "Id", "Test seed"]
    csv_content = []
    for i in range(len(set_id)):
        csv_content.append([dataset_name, labels_order[i], data_left[i], tests[i], set_id[i], test_seeds[i]])
    filename = "data/interim/{}_cross_validation_set.csv".format(dataset_name)
    save(filename, csv_header, csv_content, True)


def save_cross_validation_results(dataset_name, results):
    csv_header = ["Dataset", "Split seed", "Validation size", "Selection seed", "p", "k", "Random score",
                  "p-dispersion score", "Mahalanobis outlier detection score", "Ground truth"]
    csv_content = []
    for (ground_truth, random_scores, p_dispersion_scores, mahalanobis_scores, split_seed, selection_seeds,
         validation_sizes, p_values, k) in results:
        for i in range(len(k)):
            csv_content.append([
                dataset_name, split_seed[i], validation_sizes[i], selection_seeds[i], {p_values[i][0]:p_values[i][1]},
                k[i], random_scores[i], p_dispersion_scores[i], mahalanobis_scores[i], ground_truth[i]
            ])
    filename = "data/results/{}_cross_validation.csv".format(dataset_name)
    save(filename, csv_header, csv_content)


def save_cross_validation_test_results(dataset_name, results):
    csv_header = ["Dataset", "p", "q", "k", "Score", "Method", "Ground truth"]
    csv_content = []
    for (ground_truth, random_scores, p_dispersion_scores, hybrid_scores, selection_sizes, q, k) in results:
        csv_content.append([
            dataset_name, selection_sizes, 0, k[0], random_scores, "Random", ground_truth
        ])
        csv_content.append([
            dataset_name, selection_sizes, 0, k[1], p_dispersion_scores, "p-dispersion", ground_truth
        ])
        csv_content.append([
            dataset_name, selection_sizes, q, k[2], hybrid_scores, "Hybrid", ground_truth
        ])
    filename = "data/results/{}_cross_validation_test.csv".format(dataset_name)
    save(filename, csv_header, csv_content)


def save_p_values(dataset_name, p_values):
    csv_header = ['Dataset', 'p', 'Training size']
    csv_content = [[dataset_name, p, p_per_class]for (p, p_per_class) in p_values]
    filename = "data/interim/{}_p_values.csv".format(dataset_name)
    save(filename, csv_header, csv_content, True)
