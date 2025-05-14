from fignn.data_loader import load_relational_data
from fignn.graph_building import build_fk_graph
from fignn.augmentor import augment_graph
from fignn.models import GNN
from fignn.training import train_model, evaluate_model, log_test_metrics
from fignn.utils import assign_node_labels, split_node_labels
import os
import pandas as pd
import torch
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(model, graph, node_type, optimizer, loss_fn, tag, log_filename=None):
    print(f"### Start training: {tag} ###")
    train_model(model, graph, node_type, optimizer, loss_fn, epochs=30, verbose=True)

    print(f"### Final Test Evaluation: {tag} ###")
    test_metrics = evaluate_model(model, graph, node_type, split="test", return_metrics=True)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test ROC AUC: {test_metrics.get('roc_auc', 'N/A')}")

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        log_test_metrics(test_metrics, filename=log_filename)

    # Make sure to return the relevant metrics
    return {
        "accuracy": test_metrics.get("accuracy", float("nan")),
        "f1": test_metrics.get("f1", float("nan")),
        "roc_auc": test_metrics.get("roc_auc", float("nan"))
    }


def run_fignn(graph, db, node_id_map, labels_trainval, task_table, num_classes, k=3, method="gnn_gain"):
    # Augment graph
    aug_graph, attrs = augment_graph(
        db=db,
        initial_graph=graph,
        labels=labels_trainval,
        node_id_map=node_id_map,
        scoring_method=method,
        label_table=task_table,
        k_hops=3,
        max_attributes=k,
        threshold=0.0,
        max_depth=2,
        verbose=True,
        log_path=f"logs/augment_k{k}_{method}.csv"
    )

    # Assign labels
    assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)

    # Define model and optimizer
    model = GNN(metadata=aug_graph.metadata(), hidden_channels=64, out_channels=num_classes, target_node=task_table).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Tagging and file logging
    tag = f"FiGNN-k{k}-{method}"
    log_file = f"results/fignn_k{k}_{method}.txt"

    # Call train and evaluate
    metrics = train_and_evaluate(model, aug_graph, task_table, optimizer, loss_fn, tag=tag, log_filename=log_file)

    print("Selected attributes:", attrs)
    return metrics

if __name__ == "__main__":
    db = load_relational_data("data/synthetic")
    graph, node_id_map = build_fk_graph(db)
    available_attributes = db.get_all_attributes()

    task = pd.read_csv(os.path.join("data/synthetic", "task.csv"))
    task_table = "users"
    num_classes = 2
    labels = dict(zip(task['user_id'], task['label']))

    assign_node_labels(graph, labels, node_id_map, node_type=task_table)
    split_node_labels(graph, node_type=task_table)

    mask = graph[task_table].train_mask | graph[task_table].val_mask
    node_ids = graph[task_table]['node_ids'][mask]
    labels_trainval = {
        pk.item(): labels[pk.item()]
        for pk in node_ids
        if pk.item() in labels
    }

    results = []

    for k in range(1, 11):
        for method in ["edge_disagreement"]:  # ["mutual_info", "entropy_gain", "gnn_gain", "edge_disagreement"]: #add wl_gain
            graph_copy = copy.deepcopy(graph)
            node_id_map_copy = copy.deepcopy(node_id_map)
            metrics = run_fignn(graph_copy, db, node_id_map_copy, labels_trainval, task_table, num_classes, k=k, method=method)
            results.append({
                "method": method,
                "k": k,
                **metrics
            })

    results_df = pd.DataFrame(results)
    print(results_df)


