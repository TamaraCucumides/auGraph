from graph_building import promote_attribute
import torch.nn as nn

class IdentityMessage(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, edge_index, *args, **kwargs):
        return x


def gnn_gain_score(
    gnn_model,
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    node_id_map: dict,
    eval_fn,
    feature_dim: int,
    device=None,
    **kwargs
) -> float:
    """
    Evaluate pretrained GNN on promoted graph and compute gain in performance.

    Args:
        gnn_model: Pretrained GNN (frozen).
        graph: Original graph (HeteroData).
        db: Relational database object (to access schema).
        table: Table name where attribute lives.
        attribute: Attribute name.
        labels: dict mapping primary keys to labels.
        target_table: Table to predict on.
        node_id_map: Table → {pk → node_id}.
        eval_fn: Function to evaluate model (higher is better).
        feature_dim: Dimensionality of node features.
        device: torch.device.

    Returns:
        float: Performance gain (positive is better).
    """
    import torch
    import copy

    if device is None:
        device = next(gnn_model.parameters()).device

    # 1. Evaluate on original graph
    gnn_model.eval()
    with torch.no_grad():
        score_before = eval_fn(gnn_model, graph)

    # 2. Promote the attribute
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    # 3. Prepare the augmented graph for evaluation
    graph_aug = graph_aug.to(device)

    # 4. Make sure all nodes have features
    for node_type in graph_aug.node_types:
        if not hasattr(graph_aug[node_type], 'x') or graph_aug[node_type].x is None:
            # Create dummy features
            num_nodes = graph_aug[node_type].num_nodes
            graph_aug[node_type].x = torch.zeros((num_nodes, feature_dim), device=device)

    # 5. Handle missing edge types gracefully
    gnn_model_aug = copy.deepcopy(gnn_model)  # Work on a temp copy (safe)

    for edge_type in graph_aug.edge_types:
        if edge_type not in gnn_model_aug.message_passing_modules:
            # Patch: assign an identity message passing module
            gnn_model_aug.message_passing_modules[edge_type] = IdentityMessage()

    # 6. Evaluate on the augmented graph
    gnn_model_aug.eval()
    with torch.no_grad():
        score_after = eval_fn(gnn_model_aug, graph_aug)

    return score_after - score_before

