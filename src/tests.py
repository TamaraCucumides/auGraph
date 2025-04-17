""" from data_loader import load_relational_data
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
print(f"Mutual Information (products.category vs {task_label}): {score2:.4f}") """

""" from torch_geometric.data import HeteroData
import torch
from utils import get_label_entropies  # or wherever you place it

print("\n=== Testing get_label_entropies ===")

# Build a small test graph
data = HeteroData()

# Add 3 customer nodes with dummy features
data["customers"].x = torch.eye(3)
# Add 2 occupation nodes
data["occupation"].x = torch.eye(2)

# Connect customers to occupation
data["customers", "has_occupation", "occupation"].edge_index = torch.tensor([
    [0, 1, 2],
    [0, 1, 0]
])
data["occupation", "rev_has_occupation", "customers"].edge_index = torch.tensor([
    [0, 1, 0],
    [0, 1, 2]
])

# Labels (customer_id → label)
labels = {
    10: 0,  # customer 0
    20: 1,  # customer 1
    30: 1   # customer 2
}

# Node ID map
node_id_map = {
    "customers": {10: 0, 20: 1, 30: 2}
}

# Call the metric
entropy_result = get_label_entropies(
    data,
    target_table="customers",
    labels=labels,
    node_id_map=node_id_map,
    k_hops=2
)

print(f"→ Average neighborhood entropy: {entropy_result:.4f}") """

""" from data_loader import load_relational_data
from graph_building import build_fk_graph, promote_attribute
from scoring import label_entropy_gain
import networkx as nx

# --- 1. Load the toy relational database ---
db = load_relational_data("data/toy")

# --- 2. Define the task ---
task_table = "customers"
task_label = "high_spender"

# Fake labels for testing (primary_key → label)
labels = {
    1: 0,  # Alice
    2: 1,  # Bob
    3: 0,  # Carol
    4: 1,  # David
    5: 0   # Eve
}

# --- 3. Build schema graph (for FK path logic) ---
db.schema_graph = nx.Graph()
for src, _, dst, _ in db.foreign_keys:
    db.schema_graph.add_edge(src, dst)

# --- 4. Build the FK graph and node_id_map ---
graph, node_id_map = build_fk_graph(db)

# --- 5. Test attributes ---
table1 = "customers"
attribute1 = "occupation"

table2 = "products"
attribute2 = "category"

# --- 6. Compute and print label entropy gain ---
score1 = label_entropy_gain(
    graph=graph,
    db=db,
    table=table1,
    attribute=attribute1,
    labels=labels,
    target_table=task_table,
    node_id_map=node_id_map,
    k_hops=3
)

score2 = label_entropy_gain(
    graph=graph,
    db=db,
    table=table2,
    attribute=attribute2,
    labels=labels,
    target_table=task_table,
    node_id_map=node_id_map,
    k_hops=3
)

print(f"Label Entropy Gain (customers.occupation): {score1:.4f}")
print(f"Label Entropy Gain (products.category): {score2:.4f}") """

from data_loader import load_relational_data
from graph_building import build_fk_graph
from augmentor import augment_graph
from utils import create_random_dict

print("### Start test script ###")

# --- 1. Load the toy relational database ---
db = load_relational_data("data/synthetic")

# --- 2. Define the prediction task ---
task_table = "users" #toy is customers, synthetic is users
#task_label = "high_spender"

# Labels: primary key → label
labels = create_random_dict(200) #toy is 5, synthetic is 200

# --- 4. Build FK-based PyG graph and node index map ---
graph, node_id_map = build_fk_graph(db)

# --- 5. Run the augmentor ---
aug_graph, selected = augment_graph(
    db=db,
    initial_graph=graph,
    labels=labels,
    node_id_map=node_id_map,
    scoring_method="entropy_gain",        # "mutual_info" or "entropy_gain"
    label_table=task_table,
    #label_column=task_label,
    k_hops=2,                              # 2 * max_depth is a good rule of thumb
    max_attributes=4,
    threshold=0.00,
    max_depth=2,
    verbose=True,
    log_path="logs/augment_test.csv"
)

# --- 6. Print the result ---
print("\nSelected attributes:")
for table, attr, score in selected:
    print(f"→ {table}.{attr} (score = {score:.4f})")