from sklearn.neighbors import KNeighborsClassifier


def train_knn(training_data, training_label, n_neighbors=1):
    return KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean").fit(training_data, training_label)