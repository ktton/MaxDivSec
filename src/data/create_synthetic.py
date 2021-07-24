import numpy as np
import csv
import math
import pandas as pd
import plotly.graph_objects as go

from src.data import read_data
from src.data.save_results import save_cross_validation_sets, save_p_values

class_size = 40
filename = 'data/external/Synthetic_selection.csv'
color = ['#85C0F9', '#F5793A', '#A95AA1']
data = []
for i, (mu, sigma) in enumerate([(0, 0.5), (-3, 1), (2, 2.5)]):
    s = np.random.default_rng().normal(mu, sigma, size=(class_size, 2))
    data += [list(values) + [i] for values in s]
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['x', 'y', 'Label'])
    writer.writerows(data)

df = pd.read_csv(filename)
fig = go.Figure()
for i in df['Label'].unique():
    sub_df = df[df['Label'] == i]
    fig.add_trace(go.Scatter(x=sub_df['x'], y=sub_df['y'],  # marker_symbol='circle-open',
                             mode='markers',  marker_color=color[i]))
fig.show()

dataset = read_data.get_dataset('Synthetic', '_selection')
save_cross_validation_sets("Synthetic_selection", [[0, 1, 2]], [[list(range(0, class_size)),
                                                                 list(range(class_size, 2*class_size)),
                                                                 list(range(2*class_size, 3*class_size))]],
                           [[]], [0], [0])

p_values = [1/2, 1/4, 1/8]
p_values = [(p, int(math.ceil(3*class_size * p))) for p in p_values]
save_p_values("Synthetic_selection", p_values)
