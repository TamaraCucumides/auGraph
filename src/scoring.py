from sklearn.metrics import mutual_info_score as sklearn_mi
from utils import join_path_tables, get_label_entropies
from graph_building import promote_attribute
import networkx as nx
import copy

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
            print("Empty dataframe")
            return 0.0
        return sklearn_mi(df[attribute], df[task_label])

    # Path already guaranteed to exist
    path = nx.shortest_path(db.schema_graph, source=table, target=task_table)
    if path[0] != table:
        path = list(reversed(path))

    joined = join_path_tables(db, path)

    print("Columns in joined DataFrame:")
    print(joined.columns.tolist())

    # Assume `path` has already been computed and join has happened
    attr_col = attribute if table == path[0] else f"{table}__{attribute}"
    label_col = task_label if task_table == path[0] else f"{task_table}__{task_label}"

    print(attr_col, label_col)

    if label_col not in joined.columns or attr_col not in joined.columns:
        print("Columns not found")
        return 0.0

    df = joined[[attr_col, label_col]].dropna()
    if df.empty:
        print("Empty joined")
        return 0.0

    return sklearn_mi(df[attr_col], df[label_col])


def label_entropy_gain(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    node_id_map: dict,
    k_hops: int
) -> float:
    """
    Compute the reduction in average label entropy over k-hop neighborhoods
    of nodes in the target_table, after promoting a given attribute.

    Args:
        graph (HeteroData): Original FK-based graph.
        db (RelationalDatabase): Database for schema and data access.
        table (str): Table where the attribute is defined.
        attribute (str): Name of the attribute to promote.
        labels (dict): Mapping from primary key → label for target_table nodes.
        target_table (str): Table where the label nodes live.
        node_id_map (dict): Table → {pk → node index} mapping.
        k_hops (int): Neighborhood radius to evaluate entropy over (should be 2 * join depth).

    Returns:
        float: Average reduction in label entropy (positive is better).
    """
    # Measure before
    entropy_before = get_label_entropies(graph, target_table, labels, node_id_map, k_hops)
    print("Entropy before:", entropy_before)

    # Promote attribute and measure again
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    entropy_after = get_label_entropies(graph_aug, target_table, labels, node_id_map, k_hops)
    print("Entropy after:", entropy_after)

    return entropy_before - entropy_after

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

