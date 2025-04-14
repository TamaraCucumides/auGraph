from data_loader import load_relational_data
from scoring import mutual_info_score
import networkx as nx

# --- 1. Load the toy relational database ---
db = load_relational_data("data/toy")

# --- 2. Define the task: label is on the 'customers' table ---
task_table = "customers"
task_label = "high_spender"

# --- 3. Build schema graph (needed for join planning) ---
db.schema_graph = nx.Graph()
for src, _, dst, _ in db.foreign_keys:
    db.schema_graph.add_edge(src, dst)

# --- 4. Pick two attributes to test ---
# One from the label table (directly usable)
table1 = "customers"
attribute1 = "occupation"

# One from another table, reachable via FK path
table2 = "products"
attribute2 = "category"

# --- 5. Compute and print scores ---
score1 = mutual_info_score(
    graph=None,
    db=db,
    table=table1,
    attribute=attribute1,
    labels={},
    task_table=task_table,
    task_label=task_label
)

score2 = mutual_info_score(
    graph=None,
    db=db,
    table=table2,
    attribute=attribute2,
    labels={},
    task_table=task_table,
    task_label=task_label
)

print(f"Mutual Information (customers.occupation vs {task_label}): {score1:.4f}")
print(f"Mutual Information (products.category vs {task_label}): {score2:.4f}")