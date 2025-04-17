import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import deque
#from torch_geometric.utils import k_hop_subgraph #relevant for optimizing after, probably
import torch
from torch_geometric.data import HeteroData
import random

def create_random_dict(n, num_classes, seed=42):
    random.seed(seed)  # Set the seed for reproducibility
    return {i: random.randint(1, num_classes) for i in range(1, n + 1)}

def add_reverse_edges(data):
    """
    Adds reverse edge types to a HeteroData graph in-place.
    Skips any edge whose relation name starts with 'rev_' to avoid double reversal.
    """
    new_edges = {}
    for (src, rel, dst) in data.edge_types:
        if rel.startswith('rev_'):
            continue  # Already reversed, skip

        edge_index = data[(src, rel, dst)].edge_index
        rev_edge_index = edge_index.flip(0)
        rev_rel = f"rev_{rel}"
        new_edges[(dst, rev_rel, src)] = rev_edge_index

    for (dst, rev_rel, src), rev_edge_index in new_edges.items():
        data[(dst, rev_rel, src)].edge_index = rev_edge_index

    print(f"[add_reverse_edges] Added {len(new_edges)} reverse edge types.")
    return data

def join_path_tables(db, path):
    """
    Join tables along the given FK path in order.

    - The first table is used as the base and is NOT prefixed.
    - All other tables are prefixed, except for their join key column.
    - All FK-relevant columns are preserved across joins.

    Args:
        db (RelationalDatabase): schema and data
        path (list[str]): ordered list of table names (e.g. ['products', 'orders', 'customers'])

    Returns:
        pd.DataFrame: joined DataFrame with prefixed columns
    """
    assert len(path) >= 2, "Join path must contain at least two tables."

    joined = db.get_table(path[0]).copy()

    # Collect all FK columns that might be needed
    all_fk_keys = set()
    for src, src_col, dst, dst_col in db.foreign_keys:
        all_fk_keys.add(src_col)
        all_fk_keys.add(dst_col)

    for i in range(len(path) - 1):
        t1 = path[i]
        t2 = path[i + 1]

        # Find FK between t1 and t2
        fk = next(
            (fk for fk in db.foreign_keys if
             (fk[0] == t1 and fk[2] == t2) or
             (fk[0] == t2 and fk[2] == t1)),
            None
        )
        if fk is None:
            raise ValueError(f"No FK found between {t1} and {t2}")

        src_table, src_col, dst_table, dst_col = fk

        if t1 == src_table:
            left_key = src_col
            right_key = dst_col
            right_table = dst_table
        else:
            left_key = dst_col
            right_key = src_col
            right_table = src_table

        right_df = db.get_table(right_table).copy()

        # Rename non-key columns in right table
        right_prefix = f"{right_table}__"
        cols_to_prefix = [col for col in right_df.columns if col != right_key]
        right_renamed = right_df.rename(columns={col: right_prefix + col for col in cols_to_prefix})

        # Ensure join key is preserved
        if right_key not in right_renamed.columns:
            right_renamed[right_key] = right_df[right_key]

        # Merge
        joined = joined.merge(
            right_renamed,
            left_on=left_key,
            right_on=right_key,
            how="left"
        )

        # ✅ Preserve all FK-relevant columns for future joins
        for fk_col in all_fk_keys:
            if fk_col in right_df.columns and fk_col not in joined.columns:
                joined[fk_col] = right_df[fk_col]

    return joined


def get_label_entropies(graph, target_table, labels, node_id_map, k_hops):
    """
    Computes average label entropy for each node in the target table over its k-hop neighborhood.

    Args:
        graph (HeteroData): PyG heterogeneous graph
        target_table (str): name of the table (node type) with labels
        labels (dict): maps PK values → label (e.g., {1: 0, 2: 1, ...})
        node_id_map (dict): table → {pk → node_index}
        k_hops (int): number of hops to consider for neighborhood

    Returns:
        float: average neighborhood label entropy across all target_table nodes
    """
    if target_table not in graph.node_types:
        raise ValueError(f"Target node type '{target_table}' not found in graph.")

    index_to_label = {
        node_id_map[target_table][pk]: lbl
        for pk, lbl in labels.items()
        if pk in node_id_map[target_table]
    }

    edge_index_dict = graph.edge_index_dict
    entropies = []

    for center_node in index_to_label.keys():
        # BFS over heterogeneous edges
        visited = set([center_node])
        frontier = deque([center_node])

        for _ in range(k_hops):
            next_frontier = set()
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                # check both directions (src → dst and dst → src)
                for direction in [(0, 1), (1, 0)]:
                    src, dst = edge_index[direction[0]], edge_index[direction[1]]
                    mask = np.isin(src.numpy(), list(frontier))
                    neighbors = dst[mask].tolist()
                    next_frontier.update(neighbors)
            frontier = next_frontier - visited
            visited.update(frontier)

        # Collect labels in the k-hop neighborhood
        neighbor_labels = [index_to_label[idx] for idx in visited if idx in index_to_label]

        if not neighbor_labels:
            continue

        counts = np.bincount(neighbor_labels)
        probs = counts / counts.sum()
        ent = entropy(probs, base=2)
        entropies.append(ent)

    return float(np.mean(entropies)) if entropies else 0.0


def assign_node_labels(graph: HeteroData,
                       labels: dict,
                       node_id_map: dict,
                       node_type: str,
                       label_key: str = 'y') -> None:
    """
    Assign labels to a node type in a HeteroData object.

    Args:
        graph (HeteroData): The heterogeneous graph.
        labels (dict): A mapping {primary_key: label}.
        node_id_map (dict): Maps {table_name: {primary_key: node_idx}}.
        node_type (str): The node type to assign labels to (e.g., 'users').
        label_key (str): The field to write the labels to. Defaults to 'y'.

    Returns:
        None. Modifies the `graph` in-place by setting `graph[node_type][label_key]`.
    """
    num_nodes = graph[node_type]['x'].shape[0]
    label_tensor = torch.full((num_nodes,), -1, dtype=torch.long)

    pk_to_idx = node_id_map.get(node_type, {})
    for pk, label in labels.items():
        if pk in pk_to_idx:
            idx = pk_to_idx[pk]
            label_tensor[idx] = label

    graph[node_type][label_key] = label_tensor


def split_node_labels(graph, node_type: str, label_key: str = 'y',
                      train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                      seed=42):
    """
    Randomly splits labeled nodes into train/val/test masks.

    Modifies graph[node_type] in-place by setting train_mask, val_mask, test_mask.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    y = graph[node_type][label_key]
    labeled_idx = (y >= 0).nonzero(as_tuple=False).view(-1)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    perm = torch.randperm(len(labeled_idx))

    n_train = int(train_ratio * len(labeled_idx))
    n_val = int(val_ratio * len(labeled_idx))

    train_idx = labeled_idx[perm[:n_train]]
    val_idx = labeled_idx[perm[n_train:n_train + n_val]]
    test_idx = labeled_idx[perm[n_train + n_val:]]

    mask = torch.zeros(y.shape[0], dtype=torch.bool)
    graph[node_type]['train_mask'] = mask.clone()
    graph[node_type]['val_mask'] = mask.clone()
    graph[node_type]['test_mask'] = mask.clone()

    graph[node_type]['train_mask'][train_idx] = True
    graph[node_type]['val_mask'][val_idx] = True
    graph[node_type]['test_mask'][test_idx] = True

