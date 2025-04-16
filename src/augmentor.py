from scoring import *
from graph_building import promote_attribute

import pandas as pd
import os
from datetime import datetime

def batch_score_attributes(
    graph,
    db,
    labels,
    node_id_map,
    scoring_method: str,
    label_table: str,
    label_column: str,
    k_hops: int,
    max_depth: int = None,
    candidate_subset: list = None
):
    """
    Scores all candidate attributes (or a given subset) using the selected method.

    Returns:
        List of (score, table, attribute) sorted descending.
    """
    if candidate_subset is None:
        candidates = db.get_all_attributes(
            exclude_keys=True,
            label_table=label_table,
            max_depth=max_depth
        )
    else:
        candidates = candidate_subset

    results = []
    for table, attr in candidates:
        score = score_attribute(
            graph,
            db,
            table=table,
            attribute=attr,
            labels=labels,
            method=scoring_method,
            task_table=label_table,
            task_label=label_column,
            target_table=label_table,
            node_id_map=node_id_map,
            k_hops=k_hops
        )
        results.append((score, table, attr))

    results.sort(reverse=True)
    return results


def augment_graph(
    db,
    initial_graph,
    labels,
    node_id_map,
    scoring_method: str,
    label_table: str,
    label_column: str,
    k_hops: int = 2,
    max_attributes: int = 5,
    threshold: float = 0.0,
    max_depth: int = None,
    verbose: bool = True,
    log_path: str = None
):
    """
    Iteratively augment the graph by promoting informative attributes.

    Returns:
        final_graph: HeteroData with promoted attributes
        selected_attributes: List of (table, attribute, score) tuples
    """
    graph = initial_graph
    selected_attributes = []
    logger = []

    # Get initial candidate set
    candidates = db.get_all_attributes(
        exclude_keys=True,
        label_table=label_table,
        max_depth=max_depth
    )

    for _ in range(max_attributes):
        scored = batch_score_attributes(
            graph=graph,
            db=db,
            labels=labels,
            node_id_map=node_id_map,
            scoring_method=scoring_method,
            label_table=label_table,
            label_column=label_column,
            k_hops=k_hops,
            max_depth=max_depth,
            candidate_subset=candidates
        )

        if not scored or scored[0][0] < threshold:
            if verbose:
                print("Stopping: no attribute meets threshold.")
            break

        top_score, top_table, top_attr = scored[0]

        if verbose:
            print(f"Promoting {top_table}.{top_attr} (score = {top_score:.4f})")

        graph = promote_attribute(
            graph,
            db,
            table=top_table,
            attribute=top_attr,
            node_id_map=node_id_map,
            modify_db=True,
            inplace=False
        )

        selected_attributes.append((top_table, top_attr, top_score))
        candidates = [(t, a) for (t, a) in candidates if (t, a) != (top_table, top_attr)]

        logger.append({
            "step": len(selected_attributes),
            "table": top_table,
            "attribute": top_attr,
            "score": top_score,
            "method": scoring_method,
            "k_hops": k_hops,
            "num_node_types": len(graph.node_types),
            "num_edge_types": len(graph.edge_types)
        })

    # ✅ Handle logging to file
    if log_path:
        if log_path.endswith(".csv"):
            output_file = log_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(log_path, f"augment_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            output_file = os.path.join(log_dir, "run.csv")

        pd.DataFrame(logger).to_csv(output_file, index=False)
        if verbose:
            print(f"Log written to {output_file}")

    return graph, selected_attributes