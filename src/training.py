# training.py
import copy
import torch
import torch.nn.functional as F

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

        val_acc = evaluate_model(model, graph, node_type, split="val")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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