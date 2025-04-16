from sklearn.metrics import mutual_info_score as sklearn_mi
from utils import join_path_tables, get_label_entropies
from graph_building import promote_attribute
import networkx as nx
import pandas as pd

def score_attribute(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    method: str,
    **kwargs
) -> float:
    """
    Unified scoring interface. Dispatches to the appropriate scoring method.

    Args:
        graph: Current graph object
        db: RelationalDatabase instance
        table: Table where the attribute is defined
        attribute: Attribute name to score
        labels: Dictionary mapping PKs to labels
        method: Scoring strategy name
        kwargs: Additional arguments for specific scorers

    Returns:
        float: Score for the attribute
    """
    scorers = {
        "mutual_info": mutual_info_score,
        "entropy_gain": label_entropy_gain,
        # Add others here
        # "wl_gain": wl_refinement_gain,
        # "edge_disagreement": edge_disagreement_rate
    }

    if method not in scorers:
        raise ValueError(f"Unknown scoring method: {method}")

    return scorers[method](
        graph=graph,
        db=db,
        table=table,
        attribute=attribute,
        labels=labels,
        **kwargs
    )

def mutual_info_score(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    **kwargs
) -> float:
    """
    Compute mutual information between a candidate attribute (from `table`)
    and external labels defined over nodes in `target_table`.

    Args:
        table: Table where the attribute is defined
        attribute: Name of the attribute
        labels: Dict {primary_key → label} for nodes in `target_table`
        target_table: Table where labels are defined (i.e., the task target)

    Returns:
        Mutual information (float)
    """
    label_col = "__temp_label__"
    pk_target = db.primary_keys[target_table]

    # Create temporary label DataFrame
    label_df = pd.DataFrame([{pk_target: pk, label_col: y} for pk, y in labels.items()])

    if table == target_table:
        df = db.get_table(table)[[attribute, pk_target]].copy()
        df = df.merge(label_df, on=pk_target, how="inner")
        if df.empty:
            return 0.0
        return sklearn_mi(df[attribute], df[label_col])

    # Join path from table to target_table
    path = nx.shortest_path(db.schema_graph, source=table, target=target_table)
    if path[0] != table:
        path = list(reversed(path))

    joined = join_path_tables(db, path)

    attr_col = attribute if table == path[0] else f"{table}__{attribute}"
    target_pk_col = pk_target if target_table == path[0] else f"{target_table}__{pk_target}"

    if attr_col not in joined.columns or target_pk_col not in joined.columns:
        return 0.0

    joined[label_col] = joined[target_pk_col].map(labels)
    df = joined[[attr_col, label_col]].dropna()
    if df.empty:
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
    k_hops: int,
    **kwargs
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

