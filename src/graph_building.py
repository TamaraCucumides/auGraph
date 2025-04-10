from data_loader import RelationalDatabase
from torch_geometric.data import HeteroData
import torch
import pandas as pd

def build_fk_graph(db):
    """
    Build a heterogeneous graph from the relational database using PyTorch Geometric.

    Args:
        db (RelationalDatabase): The relational schema and data.

    Returns:
        data (HeteroData): A typed heterogeneous graph
        node_id_maps (dict): Maps table -> {pk_value: node_index}
    """
    data = HeteroData()
    node_id_maps = {}  # Maps: table_name → {primary_key_value → local node index}

    # --- Step 1: Add node types ---
    for table_name, df in db.tables.items():
        pk = db.primary_keys[table_name]
        pk_values = df[pk].tolist()

        # Build map from primary key value to node index
        id_map = {pkv: idx for idx, pkv in enumerate(pk_values)}
        node_id_maps[table_name] = id_map

        # Create dummy one-hot features, replace later with proper embedding
        num_nodes = len(df)
        data[table_name].x = torch.eye(num_nodes)
        data[table_name].node_ids = torch.tensor(pk_values)

    # --- Step 2: Add foreign key edges ---
    for src_table, src_col, dst_table, dst_col in db.foreign_keys:
        src_df = db.get_table(src_table)
        dst_map = node_id_maps[dst_table]
        src_map = node_id_maps[src_table]
        src_pk = db.primary_keys[src_table]

        src_nodes = []
        dst_nodes = []

        for _, row in src_df.iterrows():
            if pd.isna(row[src_col]) or pd.isna(row[src_pk]):
                continue
            src_id = row[src_pk]
            dst_id = row[src_col]

            if dst_id in dst_map and src_id in src_map:
                src_nodes.append(src_map[src_id])
                dst_nodes.append(dst_map[dst_id])

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_type = (src_table, f"{src_col}_to_{dst_col}", dst_table)
        data[edge_type].edge_index = edge_index

    return data, node_id_maps

def promote_attribute(graph, db: RelationalDatabase, table: str, attribute: str):
    """
    Add a new node type for attribute values and connect matching rows from the specified table.

    Args:
        graph: The existing FK graph
        db: The relational database object
        table (str): Table where attribute is defined
        attribute (str): Name of the attribute to promote

    Returns:
        Augmented graph with new nodes and edges
    """
    pass

if __name__ == "__main__":
    print("Graph Builder Test")