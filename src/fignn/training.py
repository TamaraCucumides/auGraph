# training.py
import copy
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def train_model(model, graph, node_type, optimizer, loss_fn, epochs=30, verbose=True):
    """
    Train a node classification model on a HeteroData graph, tracking best validation accuracy.

    Returns:
        model: the best model (according to validation accuracy)
    """
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x_dict, graph.edge_index_dict)
        y = graph[node_type].y
        mask = graph[node_type].train_mask
        loss = loss_fn(out[mask], y[mask])
        loss.backward()
        optimizer.step()

        metrics = evaluate_model(model, graph, node_type, split="val", return_metrics=True)
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | "
                  f"Val Acc: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


@torch.no_grad()
def evaluate_model(model, graph, node_type, split="test", return_metrics=False):
    """
    Evaluate model accuracy (and optionally F1 and ROC AUC) on a given split.

    Args:
        model: the GNN model
        graph: HeteroData with masks and labels
        node_type: the node type (e.g., 'user')
        split: one of 'train', 'val', 'test'
        return_metrics: if True, return dict with all metrics

    Returns:
        accuracy (float) if return_metrics is False
        else dict with accuracy, f1, and roc_auc
    """
    model.eval()
    out = model(graph.x_dict, graph.edge_index_dict)
    y_true = graph[node_type].y
    mask = graph[node_type][f"{split}_mask"]
    y_true = y_true[mask]
    y_pred = out[mask]
    y_pred_labels = y_pred.argmax(dim=-1)

    acc = (y_pred_labels == y_true).sum().item() / mask.sum().item()

    if not return_metrics:
        return acc

    # Compute F1 (macro for multi-class)
    f1 = f1_score(y_true.cpu(), y_pred_labels.cpu(), average='macro')

    # Compute ROC AUC 
    try:
        y_true_np = y_true.cpu().numpy()
        if y_pred.shape[1] == 2:  # Binary classification with 2 logits
            y_probs = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()  # prob for class 1
            roc_auc = roc_auc_score(y_true_np, y_probs)
        else:  # Multi-class case
            y_probs = F.softmax(y_pred, dim=1).cpu().numpy()
            roc_auc = roc_auc_score(y_true_np, y_probs, multi_class='ovr')
    except ValueError:
        roc_auc = None

    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc
    }


def log_test_metrics(metrics, filename="log.txt"):
    """
    Logs the test evaluation metrics to a text file.

    Args:
        metrics (dict): Dictionary with keys 'accuracy', 'f1', 'roc_auc'
        filename (str): File name to write to (default: 'log.txt')
    """
    with open(filename, "a") as f:
        f.write("==== Final Test Evaluation ====\n")
        f.write(f"Test Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Test F1 Score : {metrics['f1']:.4f}\n")
        if metrics['roc_auc'] is not None:
            f.write(f"Test ROC AUC  : {metrics['roc_auc']:.4f}\n")
        else:
            f.write("Test ROC AUC  : N/A (not applicable for this task)\n")
        f.write("===============================\n\n")
