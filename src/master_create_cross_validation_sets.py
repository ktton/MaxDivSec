import sys
import time

from src.data import save_results, read_data
from src.cross_validation import cross_validation

# dataset, ?categorization_mode
print('Argument List:', str(sys.argv[1:]))
# dataset_name = sys.argv[1]
for dataset_name in sys.argv[1:]:
    categorization_mode = None
    if dataset_name == 'Abalone':
        categorization_mode = '0'
    data = read_data.get_dataset(dataset_name) if categorization_mode is None \
        else read_data.get_dataset(dataset_name, categorization_mode)
    categorization_mode = "" if categorization_mode is None else categorization_mode
    dataset_name = dataset_name + categorization_mode
    if len(data) > 0:
        start_time = time.perf_counter()
        cross_validation.create_split(dataset_name, data)
        print("Created validation sets in {} for {}".format(time.perf_counter() - start_time, dataset_name))
    else:
        print("Couldn't get the data from the dataset {}".format(dataset_name))
