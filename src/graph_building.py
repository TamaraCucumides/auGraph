from data_loader import RelationalDatabase

def build_fk_graph(db: RelationalDatabase):
    """
    Build a heterogeneous graph from FK links in the relational database.

    Returns:
        A graph object (e.g., DGL or custom structure) with:
            - nodes per table
            - typed edges from FK relations
    """
    pass

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

