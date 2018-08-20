import numpy as np
import sklearn.metrics as sk_metrics


def get_best_threshold(probs, labels, threshold_min=10, threshold_max=100, interval_width=10, metric='kappa'):
    """Get the probability threshold which gives highest accuracy.
    Args:
        probs: NumPy array of probabilities.
        labels: NumPy array of labels, same shape as probabilities.
        threshold_min: Minimum value for threshold search range (units of percent).
        threshold_max: Maximum value for threshold search range (units of percent).
        interval_width: Width of interval between different probability threshold (units of percent).
        metric: The metric to determine best threshold. One of 'kappa' or 'accuracy'.
    """
    if metric != 'kappa' and metric != 'accuracy':
        raise ValueError('Expected "kappa" or "accuracy" for metric, got {}'.format(metric))
    metric_fn = sk_metrics.cohen_kappa_score if metric == 'kappa' else sk_metrics.accuracy_score

    best_metric_value = 0.
    best_threshold = 0.
    for percentage_threshold in range(threshold_min, threshold_max, interval_width):
        threshold = percentage_threshold / 100
        preds = np.array(probs) > threshold
        metric_value = metric_fn(labels, preds)
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold

    return best_threshold
