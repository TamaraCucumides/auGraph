from sklearn.metrics import mutual_info_score as sklearn_mi
from src.utils import join_path_tables
import networkx as nx

def score_attribute(graph, db, table: str, attribute: str, labels: dict, method: str) -> float:
    """
    Unified scoring interface. Dispatches to the appropriate scoring method.

    Args:
        graph: Current (possibly augmented) graph object
        db: RelationalDatabase instance
        table: Table where the attribute is defined
        attribute: Attribute name to score
        labels: Dictionary mapping node IDs to labels
        method: Scoring strategy ("mutual_info", "entropy_gain", "wl_gain", etc.)

    Returns:
        float: Score for the attribute
    """
    if method == "mutual_info":
        return mutual_info_score(graph, db, table, attribute, labels)
    elif method == "entropy_gain":
        return label_entropy_gain(graph, db, table, attribute, labels)
    elif method == "wl_gain":
        return wl_refinement_gain(graph, db, table, attribute, labels)
    elif method == "edge_disagreement":
        return edge_disagreement_rate(graph, db, table, attribute, labels)
    else:
        raise ValueError(f"Unknown scoring method: {method}")

def mutual_info_score(
    graph,db,
    table: str,
    attribute: str,
    labels: dict,
    task_table: str,
    task_label: str) -> float:
    """
    Compute mutual information between attribute and label using minimal join.
    Assumes table is FK-reachable from task_table (enforced upstream).
    """

    if table == task_table:
        df = db.get_table(table)[[attribute, task_label]].dropna()
        if df.empty:
            return 0.0
        return sklearn_mi(df[attribute], df[task_label])

    # Path already guaranteed to exist
    path = nx.shortest_path(db.schema_graph, source=table, target=task_table)
    joined = join_path_tables(db, path)

    label_col = task_label if task_table == path[-1] else f"{task_table}__{task_label}"
    attr_col = attribute if table == path[0] else f"{table}__{attribute}"

    if label_col not in joined.columns or attr_col not in joined.columns:
        return 0.0

    df = joined[[attr_col, label_col]].dropna()
    if df.empty:
        return 0.0

    return sklearn_mi(df[attr_col], df[label_col])

def label_entropy_gain(graph, db, table: str, attribute: str, labels: dict) -> float:
    """
    Simulate promoting the attribute and compute average label entropy reduction
    in local neighborhoods.
    """
    pass

def wl_refinement_gain(graph, db, table: str, attribute: str, labels: dict) -> float:
    """
    Apply 1-WL color refinement before and after promotion.
    Score is the increase in color class count.
    """
    pass

def edge_disagreement_rate(graph, db, table: str, attribute: str, labels: dict) -> float:
    """
    Compute how often promoting the attribute creates edges between nodes
    with different labels.
    """
    pass

