from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score

def eval_deep(log, loader):
    """
    Evaluate the classification performance given mini-batch data.
    """
    data_size = len(loader.dataset.indices)
    batch_size = loader.batch_size
    size_list = [batch_size] * (data_size // batch_size) + [data_size % batch_size] if data_size % batch_size else [batch_size] * (data_size // batch_size)

    assert len(log) == len(size_list)

    # Initialize metrics
    metrics = {'accuracy': 0, 'f1_macro': 0, 'f1_micro': 0, 'precision': 0, 'recall': 0}
    prob_log, label_log = [], []

    # Calculate metrics for each batch
    for (pred_prob, true_labels), size in zip(log, size_list):
        pred_labels = pred_prob.data.cpu().numpy().argmax(axis=1)
        true_labels = true_labels.data.cpu().numpy()

        prob_log.extend(pred_prob.data.cpu().numpy()[:, 1])
        label_log.extend(true_labels)

        metrics['accuracy'] += accuracy_score(true_labels, pred_labels) * size
        metrics['f1_macro'] += f1_score(true_labels, pred_labels, average='macro', zero_division=0) * size
        metrics['f1_micro'] += f1_score(true_labels, pred_labels, average='micro', zero_division=0) * size
        metrics['precision'] += precision_score(true_labels, pred_labels, zero_division=0) * size
        metrics['recall'] += recall_score(true_labels, pred_labels, zero_division=0) * size

    # Normalize by total data size
    for key in metrics:
        metrics[key] /= data_size

    metrics['auc'] = roc_auc_score(label_log, prob_log)
    metrics['ap'] = average_precision_score(label_log, prob_log)

    return metrics
