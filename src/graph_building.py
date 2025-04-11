from data_loader import load_relational_data
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

def promote_attribute(
    graph,
    db,
    table: str,
    attribute: str,
    node_id_map: dict,
    modify_db: bool = False
):
    """
    Promote a categorical attribute into a node type and connect entities to their attribute values.

    Args:
        graph (HeteroData): Current heterogeneous graph.
        db (RelationalDatabase): Database containing tables and schema.
        table (str): Table where the attribute is defined.
        attribute (str): Attribute to promote.
        node_id_map (dict): Maps each table to {primary_key_value -> node index}.
        modify_db (bool): Whether to update db with virtual table + FK.
    """
    df = db.get_table(table)
    pk = db.primary_keys[table]

    # Get all unique attribute values
    values = df[attribute].dropna().unique().tolist()
    value_to_index = {val: idx for idx, val in enumerate(values)}

    # Add new node type for attribute values
    num_values = len(values)
    attr_node_type = attribute  # node type name = attribute name
    graph[attr_node_type].x = torch.eye(num_values)  # dummy one-hot features
    graph[attr_node_type].value_strings = values     # for debugging/lookup

    # Create edges: (entity → attribute value)
    src_nodes = []
    dst_nodes = []

    for _, row in df.iterrows():
        if pd.isna(row[attribute]) or pd.isna(row[pk]):
            continue

        attr_val = row[attribute]
        entity_id = row[pk]

        if attr_val in value_to_index and entity_id in node_id_map[table]:
            src_idx = node_id_map[table][entity_id]
            dst_idx = value_to_index[attr_val]

            src_nodes.append(src_idx)
            dst_nodes.append(dst_idx)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_type = (table, f"has_{attribute}", attribute)
    graph[edge_type].edge_index = edge_index

    if modify_db:
        # Simulate schema change: new "table" for the attribute
        attr_df = pd.DataFrame({
            f"{attribute}_id": values
        })
        db.tables[attribute] = attr_df
        db.primary_keys[attribute] = f"{attribute}_id"

        # Add new "foreign key" from original table to this new virtual table
        db.foreign_keys.append((table, attribute, attribute, f"{attribute}_id"))

    return graph  # note: graph is also modified in-place

if __name__ == "__main__":
    print("Graph Builder Test")

    # First try the graph construction
    db = load_relational_data("data/toy")
    graph, id_maps = build_fk_graph(db)

    print(graph)

    # Try the attribute promotion on the graph

    graph = promote_attribute(graph, db, table="customers", attribute="occupation", node_id_map=id_maps, modify_db=True)
    print(graph)
    db.print_schema()

