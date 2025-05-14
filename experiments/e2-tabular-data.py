from ucimlrepo import fetch_ucirepo
from fignn.graph_building import build_tabular_graph, promote_attribute
from fignn.knn_graph_building import build_knn_graph
from fignn.training import train_model, evaluate_model, log_test_metrics
from fignn.utils import assign_node_labels, split_node_labels
from fignn.models import GNN
from fignn.augmentor import augment_graph
from fignn.data_loader import load_tabular_data
import torch
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def graph_from_dataframe(table, table_name, pk, type="fk"):
    build_graph = {
        "knn": build_knn_graph,
        "fk": build_tabular_graph,
    }

    if type not in build_graph:
        raise ValueError(f"Unknown graph type: {type}")

    return build_graph[type](table, table_name=table_name, pk=pk)


def train_and_evaluate(model, graph, node_type, optimizer, loss_fn, tag, log_filename=None):
    print(f"### Start training: {tag} ###")
    train_model(model, graph, node_type, optimizer, loss_fn, epochs=30, verbose=True)

    print(f"### Final Test Evaluation: {tag} ###")
    test_metrics = evaluate_model(model, graph, node_type, split="test", return_metrics=True)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test ROC AUC: {test_metrics.get('roc_auc', 'N/A')}")

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)  # Ensure folder exists
        log_test_metrics(test_metrics, filename=log_filename)

def promote_all_attributes(graph, db, node_id_map, available_attributes):
    graph_copy = graph
    for table, attribute in available_attributes:
        graph_copy = promote_attribute(
            graph_copy, db, table=table, attribute=attribute,
            node_id_map=node_id_map, modify_db=False, inplace=False
        )
    return graph_copy

def promote_random_k(graph, db, node_id_map, available_attributes, k):
    selected = random.sample(available_attributes, k)
    graph_copy = graph
    for table, attribute in selected:
        graph_copy = promote_attribute(
            graph_copy, db, table=table, attribute=attribute,
            node_id_map=node_id_map, modify_db=False, inplace=False
        )
    return graph_copy

def run_all_promoted(graph, db, task_table, node_id_map, labels, available_attributes, num_classes):
    aug_graph = promote_all_attributes(graph, db, node_id_map, available_attributes)
    assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)
    model = GNN(metadata=aug_graph.metadata(), hidden_channels=64, out_channels=num_classes, target_node=task_table).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_and_evaluate(model, aug_graph, task_table, optimizer, loss_fn, tag="All-Promoted", log_filename="results/tabular-all-promoted.txt")

def run_random_promoted(graph, db, node_id_map, task_table, labels, available_attributes, num_classes, k):
    aug_graph = promote_random_k(graph, db, node_id_map, available_attributes, k)
    assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)
    model = GNN(metadata=aug_graph.metadata(), hidden_channels=64, out_channels=num_classes, target_node=task_table).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    tag = f"Random-{k}-Promoted"
    train_and_evaluate(model, aug_graph, task_table, optimizer, loss_fn, tag=tag, log_filename=f"results/relational-random-k{k}.txt")

def run_fignn(graph, db, node_id_map, labels_trainval, task_table, num_classes, k=3):
    for method in ["mutual_info", "gnn_gain", "edge_disagreement", "entropy_gain"]: #add wl_gain
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
                max_depth=None,
                verbose=True,
                log_path=f"logs/tabular_augment_k{k}_{method}.csv"
            )
        assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)
        model = GNN(metadata=aug_graph.metadata(), hidden_channels=64, out_channels=num_classes, target_node=task_table).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        tag = f"FiGNN-k{k}-{method}"
        log_file = f"results/tabular_fignn_k{k}_{method}.txt"
        train_and_evaluate(model, aug_graph, task_table, optimizer, loss_fn, tag=tag, log_filename=log_file)
        print("Selected attributes:", attrs)

def run_knn_only(graph, task_table, num_classes):
    model = GNN(metadata=graph.metadata(), hidden_channels=64, out_channels=num_classes, target_node=task_table).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_and_evaluate(model, graph, task_table, optimizer, loss_fn, tag="KNN-only", log_filename="results/tabular-knn.txt")


if __name__ == "__main__":
    mushroom = fetch_ucirepo(id=73) 

    df_mushroom = mushroom.data.features 
    task = mushroom.data.targets 
    labels = dict(zip(task.index, task["poisonous"].map({"p": 1, "e": 0})))

    table_name = "mushroom"
    task_table = "mushroom"

    num_classes = 2



    available_attributes = [] # list of attributes available for promoting

    for col in list(mushroom.data.features.columns):
        available_attributes.append(("mushroom", col)) #(table, attribute)

    # create graph
    graph_type = "fk"
    graph, node_id_map = graph_from_dataframe(df_mushroom, table_name, pk=None, type=graph_type)

    # prepare for training
    assign_node_labels(graph, labels, node_id_map, node_type=task_table)
    split_node_labels(graph, node_type=task_table)

    mask = graph[task_table].train_mask | graph[task_table].val_mask
    node_ids = graph[task_table]['node_ids'][mask]
    labels_trainval = {
        pk.item(): labels[pk.item()]
        for pk in node_ids
        if pk.item() in labels
    }

    # train and evaluate
    if graph_type=="knn":
        run_knn_only(graph, task_table, num_classes)

    else:
        db = load_tabular_data(df_mushroom, table_name="mushroom", pk=None)

        #run_random_promoted(graph, db, node_id_map, task_table, labels, available_attributes, num_classes, k=3)
        #run_all_promoted(graph, db, task_table, node_id_map, labels, available_attributes, num_classes)
        run_fignn(graph, db, node_id_map, labels_trainval, task_table, num_classes)





