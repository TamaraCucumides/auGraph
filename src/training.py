# training.py

import torch
import torch.nn.functional as F

def train_model(model, graph, node_type, optimizer, loss_fn, epochs=30, verbose=True):
    """
    Train a node classification model on a HeteroData graph.

    Args:
        model: the GNN model
        graph: PyG HeteroData object with .y and .train_mask on `node_type`
        node_type: the node type to train on (e.g., 'users')
        optimizer: any PyTorch optimizer
        loss_fn: loss function (e.g., CrossEntropyLoss)
        epochs: number of training steps
        verbose: print progress if True
    Returns:
        model: the trained model
    """
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x_dict, graph.edge_index_dict)
        y = graph[node_type].y
        mask = graph[node_type].train_mask
        loss = loss_fn(out[mask], y[mask])
        loss.backward()
        optimizer.step()

        if verbose:
            val_acc = evaluate_model(model, graph, node_type, split="val")
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    return model


@torch.no_grad()
def evaluate_model(model, graph, node_type, split="test"):
    """
    Evaluate model accuracy on a given split.

    Args:
        model: the GNN model
        graph: HeteroData with masks and labels
        node_type: the node type (e.g., 'users')
        split: one of 'train', 'val', 'test'

    Returns:
        accuracy (float)
    """
    model.eval()
    out = model(graph.x_dict, graph.edge_index_dict)
    pred = out.argmax(dim=-1)
    y = graph[node_type].y
    mask = graph[node_type][f"{split}_mask"]
    correct = (pred[mask] == y[mask]).sum()
    total = mask.sum()
    return float(correct) / int(total)