import sys
import csv
import itertools
import pandas as pd
from scipy.stats import wilcoxon

from src.data import read_data

# dataset
print('Argument List:', str(sys.argv[1:]))
for dataset_name in sys.argv[1:]:
    print(dataset_name)
    original_random, df_random = read_data.get_cross_validation_results(dataset_name + "_euclidean_knn_random")
    df_random = df_random[df_random['Method'] == 'Random']
    random_dict = {'Accuracy': [], 'k': [], 'p': []}
    for split_seed in df_random['Split seed'].unique():
        sub_random_df = df_random[df_random['Split seed'] == split_seed]
        for p, k in itertools.product(df_random['p'].unique(), df_random['k'].unique()):
            random_dict['k'] += [k]
            random_dict['p'] += [p]
            sub_sub_random_df = sub_random_df[(sub_random_df['k'] == k) & (sub_random_df['p'] == p)]
            random_dict['Accuracy'] += [sub_sub_random_df['Accuracy'].mean()]
    random_mean_df = pd.DataFrame(data=random_dict)

    original_mahalanobis, df_mahalanobis = read_data.get_cross_validation_results(dataset_name + "_euclidean_knn_mahalanobis")
    df_mahalanobis = df_mahalanobis[df_mahalanobis['Method'] == 'Mahalanobis outlier detection']

    original_OBMA, df_OBMA = read_data.get_cross_validation_results(dataset_name + "_euclidean_mahalanobis_knn_max_diversity_OBMA")
    df_OBMA = df_OBMA[df_OBMA['Method'] == 'p-dispersion']
    df_OBMA = df_OBMA.replace('p-dispersion', 'max-diversity')

    test_results = [['Method', 'Type', 'hypothesis', 'statistic', 'p-value']]
    for method, current_df, base_df in [('rdm_maha', df_mahalanobis, random_mean_df), ('random', df_OBMA, random_mean_df), ('mahalanobis', df_OBMA, df_mahalanobis)]:
        for k in base_df['k'].unique():
            k_df = current_df[current_df['k'] == k]
            k_df_base = base_df[base_df['k'] == k]
            for p in base_df['p'].unique():
                sub_df = k_df[k_df['p'] == p]
                sub_df_base = k_df_base[k_df_base['p'] == p]
                if len(sub_df_base) != len(sub_df):
                    print("ERROR for {} {} {}".format(dataset_name, k, p))
                difference = [obma-base for obma, base in zip(sub_df['Accuracy'].values, sub_df_base['Accuracy'].values)]
                t_statistics, p_value = wilcoxon(difference, zero_method='zsplit')
                test_results.append([method, "{}-{}".format(k, p[1:].split(':')[0]), 'Different', t_statistics, p_value])
                t_statistics, p_value = wilcoxon(difference, zero_method='zsplit', alternative='greater')
                test_results.append([method, "{}-{}".format(k, p[1:].split(':')[0]), 'Better', t_statistics, p_value])
                t_statistics, p_value = wilcoxon(difference, zero_method='zsplit', alternative='less')
                test_results.append([method, "{}-{}".format(k, p[1:].split(':')[0]), 'Worst', t_statistics, p_value])

    with open("data/results/{}_wilcoxon_results.csv".format(dataset_name), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(test_results)
