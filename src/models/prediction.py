from sklearn import metrics


def predict_scores(x_test, y_test, model, metric_used):
    prediction = model.predict(x_test)
    if metric_used == 2:
        pos_label = '>50K' if '>50K' in y_test else 'b' if 'b' in y_test else 4 if 4 in y_test else 1
    else:
        pos_label = None
    accuracy_score = metrics.mean_squared_error(y_test, prediction, squared=False) if metric_used == 1 \
        else metrics.f1_score(y_test, prediction, pos_label=pos_label, average='binary') if metric_used == 2 \
        else metrics.accuracy_score(y_test, prediction)
    kappa_score = metrics.cohen_kappa_score(y_test, prediction)
    return prediction.tolist(), accuracy_score, kappa_score
