from data_loader import load_relational_data
from graph_building import build_fk_graph
from augmentor import augment_graph
from utils import create_random_dict, assign_node_labels, split_node_labels
from models import GNN
from training import train_model, evaluate_model
import torch

print("### Start test script ###")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Load relational data ---
db = load_relational_data("data/synthetic")

# --- 2. Prediction task ---
task_table = "users"
num_classes = 3
labels = create_random_dict(200, num_classes) #TODO: change this into a df

# --- 3. Build FK graph ---
graph, node_id_map = build_fk_graph(db)

# --- 3b. Assign full labels to FK-graph for splitting ---
assign_node_labels(graph, labels, node_id_map, node_type=task_table)

# --- 4. Split into train/val/test ---
split_node_labels(graph, node_type=task_table)

# --- 5. Filter labels for train+val ---
mask = graph[task_table].train_mask | graph[task_table].val_mask
node_ids = graph[task_table]['node_ids'][mask]
labels_trainval = {
    pk.item(): labels[pk.item()]
    for pk in node_ids
    if pk.item() in labels
}

# --- 6. Run augmentation (train+val only) ---
aug_graph, selected = augment_graph(
    db=db,
    initial_graph=graph,
    labels=labels_trainval,
    node_id_map=node_id_map,
    scoring_method="mutual_info",
    label_table=task_table,
    k_hops=2,
    max_attributes=4,
    threshold=0.00,
    max_depth=2,
    verbose=True,
    log_path="logs/augment_test.csv"
)

# --- 7. Assign full labels + masks to augmented graph ---
assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)
for split in ['train_mask', 'val_mask', 'test_mask']:
    aug_graph[task_table][split] = graph[task_table][split].clone()

# --- 8. Train and test GNN ---

aug_graph = aug_graph.to(device)

# Define model
model = GNN(
    metadata=aug_graph.metadata(),
    hidden_channels=64,
    out_channels=num_classes,
    target_node=task_table
).to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Train
print("### Start training ###")
train_model(
    model=model,
    graph=aug_graph,
    node_type=task_table,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    verbose=True
)

# Evaluate
print("### Final Test Evaluation ###")
test_acc = evaluate_model(model, aug_graph, node_type=task_table, split="test")
print(f"Test Accuracy: {test_acc:.4f}")

