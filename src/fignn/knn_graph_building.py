# kNN Graph Building
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors

from fignn.utils import encode_table


def build_knn_graph(table: pd.DataFrame, table_name: str = "instance", k: int = 10):
    """
    Build a k-NN graph over rows of a tabular dataset using encode_table().
    
    Args:
        table (pd.DataFrame): Input table with mixed feature types.
        table_name (str): Node type name.
        k (int): Number of neighbors for kNN.
    
    Returns:
        HeteroData: Homogeneous graph in HeteroData format.
    """
    data = HeteroData()

    # --- Step 1: Encode features using your logic ---
    x = encode_table(table)  # Already returns a torch.FloatTensor
    num_nodes = x.size(0)
    data[table_name].x = x
    data[table_name].node_ids = torch.arange(num_nodes)

    # --- Step 2: Compute kNN graph ---
    features_np = x.numpy()
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(features_np)
    knn_graph = nbrs.kneighbors(return_distance=False)

    edge_index = []
    for src, neighbors in enumerate(knn_graph):
        for dst in neighbors:
            if src != dst:
                edge_index.append([src, dst])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data[(table_name, "knn", table_name)].edge_index = edge_index

    # Optional reverse edges
    rev_index = edge_index[[1, 0]]
    data[(table_name, "rev_knn", table_name)].edge_index = rev_index

    return data