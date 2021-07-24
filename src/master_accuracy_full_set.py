import pandas as pd
import time
import numpy as np

from src.data import read_data
from src.models import train_model, prediction

accuracy_score = {'accuracy': [], 'k': [], 'dataset': [], 'split seed': []}
for dataset_name in ['Iris', 'Seed', 'Dermatology', 'Ionosphere', 'Cancer', 'Mammographic', 'Contraceptive', 'Abalone']:
    score_metric = 1 if dataset_name in ["Abalone", "Wine_quality"] else 2 if dataset_name in ["Adult", 'Ionosphere',
                                                                                               'Cancer', 'Mammographic'] else 0
    dataset = read_data.get_dataset(dataset_name, 0) if "Abalone" == dataset_name else read_data.get_dataset(
        dataset_name)
    if len(dataset) > 0:
        start_time = time.perf_counter()
        dataset_name_val = dataset_name + "0" if dataset_name == 'Abalone' else dataset_name
        validation_sets, p_values = read_data.get_cross_validation_sets(dataset_name_val)
        if len(validation_sets) > 0:
            print("Got validation sets in {}".format(time.perf_counter() - start_time))
            validation = None
            # dict_mean_training = {'3':[],'5':[], '10':[]}
            # dict_mean_predict = {'3':[],'5':[], '10':[]}
            for validation in validation_sets:
                set_id = validation['id']
                unique_labels = [int(label) for label in validation['labels']] if validation['labels'][0].isdigit() \
                    else validation['labels']
                index_per_label = [[int(i) for i in nested_array] for nested_array in validation['potential']]
                x_train = [dataset.data[i]  for nested_array in index_per_label for i in nested_array]
                y_train = []
                for label, indexes in zip(unique_labels, index_per_label):
                    y_train += [label] * len(indexes)
                x_test = [dataset.data[int(i)] for i in validation['test']]
                y_test = [dataset.target[int(i)] for i in validation['test']]
                print("prepared validation set in {}s".format(time.perf_counter() - start_time))

                for k in [3, 5, 10]:
                    start_time = time.perf_counter()
                    mahalanobis_classifier = train_model.train_knn(x_train, y_train, 'euclidean', k)
                    # dict_mean_training[str(k)] += [time.perf_counter() - start_time]
                    # start_time = time.perf_counter()
                    score = prediction.predict_scores(x_test, y_test, mahalanobis_classifier, score_metric)
                    # dict_mean_predict[str(k)] += [time.perf_counter() - start_time]
                    accuracy_score['accuracy'] += [score[1]]
                    accuracy_score['k'] += [k]
                    accuracy_score['dataset'] += [dataset_name]
                    accuracy_score['split seed'] += [set_id]
# for k in [3, 5, 10]:
#     print(k)
#     print(np.mean(dict_mean_training[str(k)]))
#     print(np.mean(dict_mean_predict[str(k)]))
df = pd.DataFrame(data=accuracy_score)
df.to_csv("data/results/full_accuracy.csv", index=False)
print("lol")
df_mean = df.groupby(['dataset', 'k'], as_index=False).mean()
print("{Dataset} & {k=3} & {k=5} & {k=10}")
for dataset in df_mean['dataset'].unique():
    df_dataset = df_mean[df_mean['dataset'] == dataset]
    print("{} & {} & {} & {}".format(dataset, df_dataset['accuracy'].values[0], df_dataset['accuracy'].values[1], df_dataset['accuracy'].values[2]))
