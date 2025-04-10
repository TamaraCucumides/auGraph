from scoring import *
from graph_building import promote_attribute


def augment_graph(db, initial_graph, labels,
scoring_fn, max_attributes=5, threshold=0.0, verbose=True):
    """
    Iteratively augment the graph by promoting informative attributes.

    Args:
        db: RelationalDatabase instance
        initial_graph: The starting FK-based graph
        labels: Dictionary of node labels
        scoring_fn: Callable taking (graph, db, table, attr, labels) -> score
        max_attributes (int): Max number of attributes to promote
        threshold (float): Minimum score to continue promotion
        verbose (bool): If True, prints status updates

    Returns:
        augmented_graph: The final augmented graph
        selected_attributes: List of (table, attr, score) tuples
    """
    graph = initial_graph.copy()
    candidates = db.get_all_attributes(exclude_keys=True)
    selected_attributes = []

    for _ in range(max_attributes):
        scores = []
        for table, attr in candidates:
            score = scoring_fn(graph, db, table, attr, labels)
            scores.append((score, table, attr))

        scores.sort(reverse=True)
        top_score, top_table, top_attr = scores[0]

        if top_score < threshold:
            if verbose:
                print(f"Stopping: best score {top_score:.4f} < threshold {threshold}")
            break

        if verbose:
            print(f"Promoting {top_table}.{top_attr} (score = {top_score:.4f})")

        graph = promote_attribute(graph, db, top_table, top_attr)  # from graph_builder
        selected_attributes.append((top_table, top_attr, top_score))
        candidates = [(t, a) for (t, a) in candidates if (t, a) != (top_table, top_attr)]

    return graph, selected_attributes