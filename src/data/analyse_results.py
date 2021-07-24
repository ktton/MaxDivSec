import itertools
import pandas as pd
import math
import os
import csv

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import read_data


def show_selection(method):
    print("showing selection for {}".format(method))
    dataset = read_data.get_dataset("Synthetic", "_selection")
    dataset_name = "Synthetic_selection"

    datas = []
    unique_labels = []
    validation_sets, p_values = read_data.get_cross_validation_sets(dataset_name)
    p_values = [str(p).replace(':', ',') for p in sorted(p_values.items())]
    filename = "data/results/{}_euclidean_mahalanobis_max_diversity_improvements.csv".format(dataset_name)\
        if 'OBMA' in method else "data/results/{}_euclidean_mahalanobis_selection.csv".format(dataset_name) \
        if method == 'mahalanobis' else None
    if filename is None:
        print("Error in method")
        return 0

    selection_df = pd.read_csv(filename)
    if 'OBMA' in method:
        selection_df = selection_df[selection_df['method'] == method]

    set_id = 0
    for validation in validation_sets:
        if validation['id'] != set_id:
            continue
        index_per_label = [[int(i) for i in nested_array] for nested_array in validation['potential']]
        datas = [[list(dataset.data[i]) for i in nested_array] for nested_array in index_per_label]
        unique_labels = [int(label) for label in validation['labels']] if validation['labels'][0].isdigit() \
            else validation['labels']
    if len(datas) > 0:
        colors = ['#85C0F9', '#F5793A', '#A95AA1']
        fig = go.Figure()
        p_values = [value for value in p_values if '0.125' in value]
        col = 1
        for p in p_values:
            current_selection = selection_df[(selection_df['p'] == str(p))
                                             & (selection_df['split seed'] == set_id)]['selection'].values

            selections = current_selection[0][2:-2].split('], [')
            selections = [[int(i.strip()) for i in idx.split(',') if i != ""] for idx in selections]
            orders = [{i: order for order, i in enumerate(idx)}
                      for idx in selections]
            selections = [sorted(selection, reverse=True) for selection in selections]

            counter = -1
            for label, data, selection, order in zip(unique_labels, datas, selections, orders):
                counter += 1
                pca_data = data.copy()
                remove_data = []
                for idx in selection:
                    remove_data.append(pca_data.pop(idx))
                pca_df = pd.DataFrame(data=pca_data, columns=('x', 'y'))

                fig.add_trace(go.Scatter(x=pca_df['x'], y=pca_df['y'], mode='markers', marker_symbol='circle-open',
                                         marker_color=colors[counter],
                                         name=str(label),
                                         legendgroup=str(label)),)
                remove_df = pd.DataFrame(data=remove_data, columns=('x', 'y'))
                fig.add_trace(go.Scatter(x=remove_df['x'], y=remove_df['y'], mode='markers',
                                         marker_color=colors[counter],
                                         name=str(label),
                                         legendgroup=str(label),
                                         showlegend=False))
            col += 1

        fig.update_xaxes(matches=None, showticklabels=True,
                         zeroline=False, showline=True,  linewidth=2, linecolor='black', ticks='outside',
                         range=[-6, 2], dtick=2)
        fig.update_yaxes(matches=None,  showticklabels=True,
                         zeroline=False, showline=True,  linewidth=2, linecolor='black', ticks='inside',
                         range=[-6, 2], dtick=2)
        fig.update_traces(marker_line_width=2, marker_size=10)
        fig.update_layout(height=550, width=550)
        fig.update_traces(showlegend=False)

        fig.write_image("figures/{}_{}_selection.eps".format(dataset_name, method))
        fig.write_image("figures/{}_{}_selection.svg".format(dataset_name, method))
        fig.show()
    else:
        print("couldn't get data for validation set")


def analyse_wilcoxon_tests():
    thresholds = [0.01, 0.05]
    for threshold in thresholds:
        all_dataset = ['Iris', 'Seed', 'Dermatology', 'Ionosphere', 'Cancer', 'Mammographic', 'Contraceptive', 'Abalone0']
        result_rows = [['Dataset', 'Method', 'Test', 'Result']]
        for dataset in all_dataset:
            use_RMSE = dataset == 'Abalone0'  #Lower RMSE score are better so we need to invert the logic
            path = "data/results/{}_wilcoxon_results.csv".format(dataset)
            if os.path.exists(path):
                all_results_dict = {'Different': {'p < 0.01': [], 'p > 0.01': []},
                                    'Better': {'p < 0.01': [], 'p > 0.01': []},
                                    'Worst': {'p < 0.01': [], 'p > 0.01': []}}
                df = pd.read_csv(path)
                sub_better_df = df[df['hypothesis'] == 'Different']
                for k in [3, 5, 10]:
                    for p in [0.5, 0.25, 0.125]:
                        sub_sub_df = sub_better_df[sub_better_df['Type'] == '{}-{}'.format(k, p)]
                        for row in sub_sub_df.itertuples():
                            current_test = [[row[1], row[2]]]
                            key = 'p < 0.01' if row[-1] < threshold else 'p > 0.01'
                            all_results_dict['Different'][key] += current_test
                for key, combinations in all_results_dict['Different'].items():
                    different = 1 if key == 'p < 0.01' else 0
                    sub_better_df = df[df['hypothesis'] == 'Better']
                    sub_worst_df = df[df['hypothesis'] == 'Worst']
                    for combination in combinations:
                        result_rows.append([dataset, combination, 'Different', different])
                        if different == 1:
                            hypothesis_array = [(sub_better_df, 'Worst'), (sub_worst_df, 'Better')] if use_RMSE \
                                else [(sub_better_df, 'Better'), (sub_worst_df, 'Worst')]
                            for hypothesis_df, hypothesis in hypothesis_array:
                                sub_sub_df = hypothesis_df[(hypothesis_df['Type'] == combination[1]) & (
                                        hypothesis_df['Method'] == combination[0])]
                                if len(sub_sub_df) > 1:
                                    print("Error")
                                    continue
                                else:
                                    rejected = 1 if sub_sub_df['p-value'].values[0] < threshold else 0
                                    new_key = 'p < 0.01' if rejected == 1 else 'p > 0.01'
                                    all_results_dict[hypothesis][new_key] += [combination]
                                    result_rows.append([dataset, combination, hypothesis, rejected])
                print(dataset)
                print(all_results_dict)
                for hypothesis, results in all_results_dict.items():
                    print(hypothesis)
                    print([(key, len(results[key])) for key in results.keys()])
                print('done')
            else:
                print("Missing file for {}".format(dataset))
        if len(result_rows) > 1:
            number = str(threshold)[-1]
            with open("data/results/Analysed_{}_wilcoxon_results.csv".format(number), 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(result_rows)
            read_analysed_wilcoxon_tests(number)


def read_analysed_wilcoxon_tests(number):
    color = ['#85C0F9', '#F5793A', '#A95AA1']
    df = pd.read_csv("data/results/Analysed_{}_wilcoxon_results.csv".format(number))
    all_dataset = ['Iris', 'Seed', 'Dermatology', 'Ionosphere', 'Cancer', 'Mammographic', 'Contraceptive', 'Abalone0']
    print("0.0{}".format(number))
    csv_rows = [['Dataset', 'Different_Random', 'Better_Random', 'Maha_Different_Random', 'Maha_Better_Random', 'Different_Maha', 'Better_Maha']]
    better_random = []
    maha_better_random = []
    better_mahalanobis = []
    for dataset in all_dataset:
        print(dataset)
        new_row = [dataset]
        dataset_df = df[df['Dataset'] == dataset]
        for method, better_array in [('Random', better_random), ('Rdm_maha', maha_better_random), ('Mahalanobis', better_mahalanobis)]:
            method_df = dataset_df.loc[dataset_df['Method'].str.contains(method, case=False)]

            different_df = method_df[method_df['Test'] == 'Different']
            result_df = different_df[different_df['Result'] == 1]
            new_row += ["{}/{}".format(len(result_df), len(different_df))]
            print(['{} that are different'.format(method), len(result_df), len(different_df)])

            different_df = method_df[method_df['Test'] == 'Better']
            result_df = different_df[different_df['Result'] == 1]
            if len(result_df) > 0:
                better_array += [(dataset, value) for value in result_df['Method'].values]
            new_row += ["{}/{}".format(len(result_df), len(different_df))]
            print(['{} that are better'.format(method), len(result_df), len(different_df)])

        csv_rows.append(new_row)
        with open("data/results/Table_{}_wilcoxon_results.csv".format(number), 'w', newline='') as f:
            writer = csv.writer(f, delimiter='&')
            writer.writerows(csv_rows)

    dict_better = {'Dataset': [], 'k': [], 'α': [], 'Method': []}
    for better_array, method_name in [(better_random, 'random'), (better_mahalanobis, 'mahalanobis')]:
        if method_name == 'mahalanobis':
            continue
        for dataset, method in better_array:
            dict_better['Method'] += [method_name]
            method_array = eval(method)[1].split('-')
            dict_better['Dataset'] += [dataset]
            dict_better['k'] += [int(method_array[0])]
            dict_better['α'] += [method_array[1]]
    df_better = pd.DataFrame(data=dict_better)
    fig = px.histogram(df_better, x="k", category_orders={'k': sorted(df_better['k'].unique())}, color='Method',
                       color_discrete_sequence=color)
    for trace in fig['data']:
        trace['name'] = trace['name'].replace("Method=", "")
    fig.update_xaxes(type='category')
    fig.update_yaxes(range=[0, 26])
    fig.add_shape(type="line", xref='paper', yref='y', x0=0, x1=0.98, y0=24, y1=24)
    fig.update_layout(
        barmode='group',
        height=1000,
        width=2000,
        font_family="Times New Roman",
        font=dict(
            size=65,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        showlegend=False
    )
    fig.add_annotation(x=0.65, y=22, xref="paper", yref='y', showarrow=False, text="Maximal number of configurations",
                       font=dict(size=60))
    fig.write_image("figures/comparative_histogram_by_k.eps")
    fig.write_image("figures/comparative_histogram_by_k.svg")
    fig = px.histogram(df_better, x="α", category_orders={'α': sorted(df_better['α'].unique(), reverse=True)},
                       color='Method', color_discrete_sequence=color)
    for trace in fig['data']:
        trace['name'] = trace['name'].replace("Method=", "")
    fig.update_xaxes(type='category')
    fig.update_yaxes(range=[0, 26])
    fig.add_shape(type="line", xref='paper', yref='y', x0=0, x1=0.98, y0=24, y1=24)
    fig.update_layout(
        barmode='group',
        height=1000,
        width=2000,
        font_family="Times New Roman",
        font=dict(
            size=65,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        ),
        showlegend=False
    )
    fig.add_annotation(x=0.65, y=22, xref="paper", yref='y', showarrow=False, text="Maximal number of configurations",
                       font=dict(size=60))
    fig.write_image("figures/comparative_histogram_by_p.eps")
    fig.write_image("figures/comparative_histogram_by_p.svg")


def show_mean_knn_cross_validation_results(dataset_name):
    color = ['#85C0F9', '#F5793A', '#A95AA1']
    use_rmse = any(name in dataset_name for name in ["Abalone", "Wine_quality"])
    use_F1 = any(name in dataset_name for name in ["Adult", 'Ionosphere', 'Cancer', 'Mammographic'])
    score_name = 'RMSE' if use_rmse else "F1-score" if use_F1 else "Accuracy"

    original_data, df = read_data.get_cross_validation_results(dataset_name + "_knn_random")
    df = df[df['Method'] == 'Random']

    original_data, df_mahalanobis = read_data.get_cross_validation_results(dataset_name + "_knn_mahalanobis")
    df_mahalanobis = df_mahalanobis[df_mahalanobis['Method'] == 'Mahalanobis outlier detection']
    df = df.append(df_mahalanobis).reset_index(drop=True)

    p_values_df = pd.read_csv("data/interim/{}_p_values.csv".format(dataset_name.split('_')[0]))
    category_order = [(1 - p) * 100 for p in sorted(p_values_df['p'].values, reverse=True)]

    original_data_max, df_max = read_data.get_cross_validation_results(
        dataset_name + "_mahalanobis_knn_max_diversity_OBMA")
    df_max = df_max.replace("p-dispersion", "MaxDivSec")
    df = df.append(df_max).reset_index(drop=True)
    results = df

    if 'Abalone' in dataset_name:
        dataset_name = "Abalone_euclidean"
    p_values = sorted(original_data['p'].unique())
    p_title = ["Training size: {}% ({})".format(100 * (1 - list(eval(p).keys())[0]), list(eval(p).values())[0]) for p in
               p_values]

    n_col_p = 3
    n_row_p = int(math.ceil(len(p_values) / n_col_p))

    method_order = ['Random', 'Mahalanobis outlier detection', 'MaxDivSec']

    fig = make_subplots(rows=n_row_p, cols=n_col_p, subplot_titles=p_title, vertical_spacing=0.1,
                        horizontal_spacing=0.05,
                        x_title="<b>k</b>", y_title="<b>F1-score</b>".format(score_name) if use_F1
            else "<b>{} score</b>".format(score_name))
    for i, p in enumerate(p_values):
        p_df = results[results['p'] == p]

        hide_legend = i == len(p_values) - 2
        row = int(i / n_col_p) + 1
        column = (i % n_col_p) + 1

        sub_fig = px.box(p_df, x="k", y="Accuracy", color="Method", points="all",
                         color_discrete_sequence=color,
                         category_orders={"k": sorted(results['k'].unique()), "Method": method_order})
        for trace in sub_fig['data']:
            trace['name'] = trace['name'].replace("Method=","")
            fig.add_trace(trace, row, column)
        if hide_legend:
            fig.update_traces(showlegend=False)

    for annotation in fig['layout']['annotations']:
        font = annotation['font']
        font['size'] = 30
        annotation['font'] = font
    fig['layout']['annotations'][-1]['xshift'] = -75  # Moves the y axis title a little bit to the left
    fig.add_annotation(dict(font=dict(color="black", size=24),
                            x=0,
                            y=1.1,
                            showarrow=False,
                            text='<b>Lower score is better</b>' if use_rmse else "<b>Higher score is better</b>",
                            textangle=0,
                            xref="x",
                            yref="paper"
                            ))
    fig.update_layout(
        height=1000,
        width=2010,
        font=dict(
            size=24,
        ),
        margin=dict(l=125, t=100),
        boxmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=30)
        )
    )
    fig.update_yaxes(matches='y')
    fig.update_xaxes(type='category')
    fig.write_image("figures/{}_cross_validation.eps".format(dataset_name))
    fig.write_image("docs/img/{}_cross_validation.svg".format(dataset_name))

    k_values = results['k'].unique()
    classifier_title = ["{}NN classifier".format(i) for i in k_values]

    n_col_k = 3
    n_row_k = int(math.ceil(len(k_values) / n_col_k))

    fig = make_subplots(rows=n_row_k, cols=n_col_k, subplot_titles=classifier_title, vertical_spacing=0.1,
                        horizontal_spacing=0.05,
                        x_title="<b>Training size (%) </b>", y_title="<b>F1-score</b>".format(score_name) if use_F1
            else "<b>{} score</b>".format(score_name))
    for i, k in enumerate(k_values):
        k_df = results[results['k'] == k].copy()
        k_df['p'] = k_df['p'].apply(lambda x: 100 * (1 - float(x[1:-1].split(':')[0])))

        hide_legend = i == len(k_values) - 2
        row = int(i / n_col_k) + 1
        column = (i % n_col_k) + 1

        k_df = k_df.sort_values(by=['p'], ascending=False)
        sub_fig = px.box(k_df, x="p", y="Accuracy", color="Method", points="all",
                         color_discrete_sequence=color,
                         category_orders={"p": category_order, "Method": method_order})
        for trace in sub_fig['data']:
            trace['name'] = trace['name'].replace("Method=", "")
            fig.add_trace(trace, row, column)
        if hide_legend:
            fig.update_traces(showlegend=False)

    for annotation in fig['layout']['annotations']:
        font = annotation['font']
        font['size'] = 30
        annotation['font'] = font
    fig['layout']['annotations'][-1]['xshift'] = -75  # Moves the y axis title a little bit to the left
    fig.add_annotation(dict(font=dict(color="black", size=24),
                            x=0,
                            y=1.1,
                            showarrow=False,
                            text='<b>Lower score is better</b>' if use_rmse else "<b>Higher score is better</b>",
                            textangle=0,
                            xref="x",
                            yref="paper"
                            ))
    fig.update_traces(boxmean=True)
    fig.update_layout(
        height=1000,
        width=2010,
        font=dict(
            size=24
        ),
        margin=dict(l=125, t=100),
        boxmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            font=dict(size=30)
        )
    )
    fig.update_yaxes(matches='y')
    fig.update_xaxes(type='category')
    fig.write_image("figures/{}_cross_validation_knn.eps".format(dataset_name))
    fig.write_image("docs/img/{}_cross_validation_knn.svg".format(dataset_name))


def get_mean_results(is_abalone=False):
    all_dataset = ['Abalone'] if is_abalone else ['Iris', 'Seed', 'Dermatology', 'Ionosphere', 'Cancer',
                                                  'Mammographic', 'Contraceptive']
    color = ['#85C0F9', '#F5793A', '#A95AA1']
    print(all_dataset)
    accuracy_df = pd.read_csv("data/results/full_accuracy.csv")

    for dataset_name in all_dataset:
        full_accuracy_df = accuracy_df[accuracy_df['dataset'] == dataset_name].copy()
        full_accuracy_df = full_accuracy_df.groupby('k', as_index=False).mean()

        use_rmse = any(name in dataset_name for name in ["Abalone", "Wine_quality"])
        use_F1 = any(name in dataset_name for name in ["Adult", 'Ionosphere', 'Cancer', 'Mammographic'])
        score_name = 'RMSE' if use_rmse else "F1-score" if use_F1 else "Accuracy"

        dataset_name_file = dataset_name + '0' if dataset_name == 'Abalone' else dataset_name

        original_data, df_random = read_data.get_cross_validation_results(dataset_name_file + "_euclidean_knn_random")
        original_data, df_mahalanobis = read_data.get_cross_validation_results(
            dataset_name_file + "_euclidean_knn_mahalanobis")
        original_data, df_obma = read_data.get_cross_validation_results(
            dataset_name_file + "_euclidean_mahalanobis_knn_max_diversity_OBMA")

        df_mahalanobis = df_mahalanobis[df_mahalanobis['Selection seed'] == 0]
        df_mahalanobis = df_mahalanobis[df_mahalanobis['Method'] == 'Mahalanobis outlier detection']
        df_obma = df_obma[df_obma['Method'] == 'p-dispersion']

        df_random = df_random[df_random['Method'] == 'Random']
        random_dict = {'Accuracy': [], 'k': [], 'p': []}
        for p, k in itertools.product(df_random['p'].unique(), df_random['k'].unique()):
            random_dict['k'] += [k]
            random_dict['p'] += [p]
            sub_sub_random_df = df_random[(df_random['k'] == k) & (df_random['p'] == p)]
            random_dict['Accuracy'] += [sub_sub_random_df['Accuracy'].mean()]
        df_random = pd.DataFrame(data=random_dict)

        random_accuracy = []
        mahalanobis_accuracy = []
        obma_accuracy = []
        p_values = sorted(original_data[~original_data['p'].str.contains("0.75")]['p'].unique())
        p_title = ["Training size: {}% ({})".format(100 * (1 - list(eval(p).keys())[0]), list(eval(p).values())[0]) for
                   p in p_values]
        fig = make_subplots(rows=1, cols=3, subplot_titles=p_title, vertical_spacing=0.1,
                            horizontal_spacing=0.05,
                            x_title="<b>k</b>", y_title="<b>F1-score</b>".format(score_name) if use_F1
            else "<b>{} score</b>".format(score_name))
        fig.update_layout(
            height=1000,
            width=2010,
            font=dict(
                size=24
            ),
            margin=dict(l=125, t=100),
            boxmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1
            )
        )
        fig.update_yaxes(matches='y')
        for i, p in enumerate(p_values):
            fig.update_traces(showlegend=False)
            p_df_random = df_random[df_random['p'] == p]
            p_df_mahalanobis = df_mahalanobis[df_mahalanobis['p'] == p]
            p_df_obma = df_obma[df_obma['p'] == p]

            mean_df_random = p_df_random.groupby('k', as_index=False).mean()
            mean_df_mahalanobis = p_df_mahalanobis.groupby('k', as_index=False).mean()
            mean_df_obma = p_df_obma.groupby('k', as_index=False).mean()

            random_accuracy += list(mean_df_random['Accuracy'].values)
            mahalanobis_accuracy += list(mean_df_mahalanobis['Accuracy'].values)
            obma_accuracy += list(mean_df_obma['Accuracy'].values)

            fig.add_trace(go.Scatter(x=mean_df_random['k'].values, y=mean_df_random['Accuracy'].values,
                                     mode='lines+markers', name='Random',
                                     marker=dict(
                                         color=color[0],
                                         size=20
                                     )), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=mean_df_mahalanobis['k'].values, y=mean_df_mahalanobis['Accuracy'].values,
                                     mode='lines+markers', name='Mahalanobis outlier detection',
                                     marker=dict(
                                         color=color[1],
                                         size=20
                                     )), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=mean_df_obma['k'].values, y=mean_df_obma['Accuracy'].values,
                                     mode='lines+markers', name='MaxDivSec',
                                     marker=dict(
                                         color=color[2],
                                         size=20
                                     )), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=full_accuracy_df['k'], y=full_accuracy_df['accuracy'],
                                     mode='lines', name='Benchmark performance',
                                     line=dict(
                                         color='black',
                                         width=4,
                                         dash='dash'
                                     )), row=1, col=i + 1)

        for annotation in fig['layout']['annotations']:
            font = annotation['font']
            font['size'] = 30
            annotation['font'] = font
        fig['layout']['annotations'][-1]['xshift'] = -75  # Moves the y axis title a little bit to the left
        fig.add_annotation(dict(font=dict(color="black", size=24),
                                # x=x_loc,
                                x=0,
                                y=1.1,
                                showarrow=False,
                                text='<b>Lower score is better</b>' if use_rmse else "<b>Higher score is better</b>",
                                textangle=0,
                                xref="x",
                                yref="paper"
                                ))

        max_accuracy = max([max(random_accuracy), max(mahalanobis_accuracy),
                            max(obma_accuracy)])
        min_accuracy = min([min(random_accuracy), min(mahalanobis_accuracy),
                            min(obma_accuracy)])
        if max_accuracy - min_accuracy < 0.1:
            fig.update_yaxes(range=[min_accuracy - 0.05, max_accuracy+0.05])
        fig.update_xaxes(type='category')
        fig.write_image("figures/{}_euclidean_mean_cross_validation.eps".format(dataset_name))
        fig.write_image("docs/img/{}_euclidean_mean_cross_validation.svg".format(dataset_name))

        k_values = original_data['k'].unique()
        classifier_title = ["{}NN classifier".format(i) for i in k_values]
        category_order = [100 * (1 - list(eval(p).keys())[0]) for p in sorted(p_values)]
        fig = make_subplots(rows=1, cols=3, subplot_titles=classifier_title, vertical_spacing=0.1,
                            horizontal_spacing=0.05,
                            x_title="<b>Training size (%) </b>", y_title="<b>F1-score</b>".format(score_name) if use_F1
            else "<b>{} score</b>".format(score_name))
        fig.update_layout(
            height=1000,
            width=2010,
            font=dict(
                size=24
            ),
            margin=dict(t=100, l=125),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1
            )
        )
        fig.update_yaxes(matches='y')
        random_accuracy = []
        mahalanobis_accuracy = []
        obma_accuracy = []
        for i, k in enumerate(k_values):
            fig.update_traces(showlegend=False)
            k_df_random = df_random[df_random['k'] == k].copy()
            k_df_random['p'] = k_df_random['p'].apply(lambda x: 100 * (1 - float(x[1:-1].split(':')[0])))
            k_df_mahalanobis = df_mahalanobis[df_mahalanobis['k'] == k].copy()
            k_df_mahalanobis['p'] = k_df_mahalanobis['p'].apply(lambda x: 100 * (1 - float(x[1:-1].split(':')[0])))
            k_df_obma = df_obma[df_obma['k'] == k].copy()
            k_df_obma['p'] = k_df_obma['p'].apply(lambda x: 100 * (1 - float(x[1:-1].split(':')[0])))

            mean_df_random = k_df_random.groupby('p', as_index=False).mean()
            mean_df_mahalanobis = k_df_mahalanobis.groupby('p', as_index=False).mean()
            mean_df_obma = k_df_obma.groupby('p', as_index=False).mean()

            random_accuracy += list(mean_df_random['Accuracy'].values)
            mahalanobis_accuracy += list(mean_df_mahalanobis['Accuracy'].values)
            obma_accuracy += list(mean_df_obma['Accuracy'].values)

            fig.add_trace(go.Scatter(x=mean_df_random['p'].values, y=mean_df_random['Accuracy'].values,
                                     mode='lines+markers', name='Random',
                                     marker=dict(
                                         color=color[0],
                                         size=20
                                     )), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=mean_df_mahalanobis['p'].values, y=mean_df_mahalanobis['Accuracy'].values,
                                     mode='lines+markers', name='Mahalanobis outlier detection',
                                     marker=dict(
                                         color=color[1],
                                         size=20
                                     )), row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=mean_df_obma['p'].values, y=mean_df_obma['Accuracy'].values,
                                     mode='lines+markers', name='MaxDivSec',
                                     marker=dict(
                                         color=color[2],
                                         size=20
                                     )), row=1, col=i + 1)
            sub_df = full_accuracy_df[full_accuracy_df['k'] == k]
            fig.add_trace(go.Scatter(x=mean_df_obma['p'].values,
                                     y=[sub_df['accuracy'].values[0]] * len(mean_df_obma['p'].unique()),
                                     mode='lines', name='Benchmark performance',
                                     line=dict(
                                         color='black',
                                         width = 4,
                                         dash='dash'
                                     )), row=1, col=i + 1)
        for annotation in fig['layout']['annotations']:
            font = annotation['font']
            font['size'] = 30
            annotation['font'] = font
        fig['layout']['annotations'][-1]['xshift'] = -75  # Moves the y axis title a little bit to the left
        fig.add_annotation(dict(font=dict(color="black", size=24),
                                # x=x_loc,
                                x=0,
                                y=1.1,
                                showarrow=False,
                                text='<b>Lower score is better</b>' if use_rmse else "<b>Higher score is better</b>",
                                textangle=0,
                                xref="x",
                                yref="paper"
                                ))
        max_accuracy = max([max(random_accuracy), max(mahalanobis_accuracy),
                            max(obma_accuracy)])
        min_accuracy = min([min(random_accuracy), min(mahalanobis_accuracy),
                            min(obma_accuracy)])
        if max_accuracy - min_accuracy < 0.1:
            fig.update_yaxes(range=[min_accuracy - 0.05, max_accuracy + 0.05])
        fig.update_xaxes(type='category', categoryorder='array', categoryarray=category_order)
        fig.write_image("figures/{}_euclidean_mean_cross_validation_knn.eps".format(dataset_name))
        fig.write_image("docs/img/{}_euclidean_mean_cross_validation_knn.svg".format(dataset_name))


# Creation of figures and tables
show_mean_knn_cross_validation_results("Iris_euclidean")
show_mean_knn_cross_validation_results("Seed_euclidean")
show_mean_knn_cross_validation_results("Ionosphere_euclidean")
show_mean_knn_cross_validation_results("Cancer_euclidean")
show_mean_knn_cross_validation_results("Abalone0_euclidean")
show_mean_knn_cross_validation_results("Contraceptive_euclidean")
show_mean_knn_cross_validation_results("Mammographic_euclidean")
show_mean_knn_cross_validation_results("Dermatology_euclidean")
get_mean_results()
get_mean_results(is_abalone=True)

analyse_wilcoxon_tests()
read_analysed_wilcoxon_tests(1)
read_analysed_wilcoxon_tests(5)

show_selection('OBMA')
show_selection('mahalanobis')
