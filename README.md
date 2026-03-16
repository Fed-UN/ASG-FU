# ASG-FU: An Adaptive and Secure Grouping Framework for Federated Unlearning

This repository contains the source code for our paper **“ASG-FU: An Adaptive and Secure Grouping Framework for Federated Unlearning”** . The project implements a federated learning system that dynamically groups clients based on model similarity, trains subgroups in parallel, and supports efficient unlearning by removing a client from its group and retraining only the affected subgroup.

## Overview

The system consists of three main phases:

1. **Initial global training** – A short federated round involving all clients to collect initial local models.
2. **Clustering & grouping** – Client models are converted to feature vectors, clustered using K‑Means, and groups are balanced via an intelligent rebalancing algorithm.
3. **Subgroup parallel training** – Each group runs its own federated learning process (with optional malicious attack simulation, early stopping, and model rollback).
4. **Global aggregation** – Group models are aggregated using a weighted combination of accuracy and data size.
5. **Unlearning** – A client is removed from a randomly selected group; the group’s model is re‑aggregated from the remaining clients, and the group is retrained. The final global model is updated accordingly.

All components are built on [Flower](https://flower.dev/) (v1.14.0) and use PyTorch for model training.

## Repository Structure

```
.
├── attack_logs/                     # JSON logs of malicious attacks (per group, per round)
├── clients/
│   ├── group/                        # Template for a client group (copy for each group)
│   │   ├── client_app.py
│   │   ├── server_app.py
│   │   ├── task.py
│   │   └── pyproject.toml
│   ├── Group_0/ ... Group_3/         # Actual groups created during execution
│   │   ├── local_models/             # Client‑side model checkpoints
│   │   └── ... (copied from template)
├── data/
│   ├── cifar-10-batches-py/
│   └── FashionMNIST/
├── global-server/
│   ├── data_partitions/               # Dirichlet split indices for each client
│   ├── global_models/                  # Global models from each group (per round)
│   ├── grouped_local_models/           # Aggregated group models after clustering
│   ├── group_results/                   # round1_clusters.json – clustering output
│   ├── local_models/                     # Client models from initial global round
│   ├── client_app.py
│   ├── server_app.py
│   ├── task.py
│   └── pyproject.toml
├── training_logs/                     # CSV logs for global and per‑group accuracy
├── training_monitor/                   # Synchronisation files for group completion
└── main.py                             # Main entry script
```

## Requirements

- Python 3.10 or 3.11 recommended
- Dependencies:
  - `flwr[simulation]=1.14.0`
  - `torch==2.10.1`
  - `torchvision==0.25.0`
  - `scikit-learn`
  - `numpy`
  - `scipy`

You can install all dependencies with:

```bash
pip install flwr[simulation] torch==2.10.1 torchvision==0.25.0 scikit-learn numpy scipy
```

## Configuration

Key parameters are set in the `pyproject.toml` files:

- **Global server** (`global-server/pyproject.toml`):
  - `num-server-rounds = 1` (only one round for initial global training)
  - `num-supernodes = 20` (total number of clients)

- **Group server** (`clients/group/pyproject.toml`):
  - `num-server-rounds = 100` (maximum rounds per group, early stopping may stop earlier)
  - `fraction-fit = 1.0`
  - `local-epochs = 5`
  - `early-stopping-rounds = 3`
  - `min-delta = 0.002`
  - `max-delta = 0.01`
  - `attack-enable`, `attack-start`, `attack-end`, `attack-type`, `attack-strength` (to simulate malicious server behaviour)
  - `rollback-enabled = true`

- **Main script** (`main.py`):
  - `unlearn_ratio = 0.1` – proportion of groups to unlearn from

## Running the Experiment

Simply execute the main script from the project root:

```bash
python main.py
```

The script will:

1. Clean up any previous runs (remove Ray temp files, old client groups, logs, models).
2. Run the global server (one round of FL) to obtain initial client models.
3. Cluster clients into groups (the number of groups is determined automatically via silhouette score and size variance).
4. Create per‑group directories and start parallel training for all groups.
5. Wait for all groups to finish (or early‑stop) and then perform global aggregation.
6. Execute unlearning: pick one group, remove one client, re‑aggregate that group’s model from the remaining clients, and retrain only that group.
7. Perform a second global aggregation with the updated group.

Logs and model checkpoints are saved in `training_logs/`, `global-server/global_models/`, and `clients/Group_*/local_models/`.

## Key Components Explained

### Clustering and Group Balancing (`main.py:perform_grouping`)

- Client models from round 1 are converted to vectors.
- The optimal number of clusters `k` is chosen by combining the silhouette score, cluster size variance, and a penalty for deviation from √n.
- Initial K‑Means labels are refined by an `intelligent_rebalance` function that ensures each group has roughly equal size (based on `target_size = n_clients // k`).
- For each group, a **similarity matrix** is computed, and the client with the highest average similarity (to others) is marked as the **aggregator** (used only for selection logic, not for actual aggregation).

### Group Training (`clients/group/server_app.py`)

Each group runs its own Flower server with a custom strategy (`MaliciousEarlyStoppingStrategy`) that supports:

- **Malicious attacks**: noise injection, negative scaling, model replacement, dropout (configurable via `pyproject.toml`).
- **Early stopping**: stops training if validation loss does not improve for a given number of rounds.
- **Model rollback**: if accuracy declines for two consecutive rounds, the server rolls back to an earlier checkpoint (up to 3 rollbacks).
- **Similarity logging**: every 3 rounds, the server computes cosine similarity among client models and saves the result.

### Global Aggregation (`main.py:global_aggregation`)

After all groups finish, the latest model of each group is evaluated on the global test set.  
Weights for aggregation are computed as:

```
w_i = 0.7 * (softmax(5 * acc_i)) + 0.3 * (data_size_i / total_data)
```

The final global model is the weighted average of the group models.

### Unlearning (`main.py:unlearning`)

- One group is selected randomly.
- One client is removed from that group’s member list.
- The remaining clients’ round‑1 models are averaged to form a new group model (`reaggregate_group_models`).
- The group is then retrained (the script re‑runs parallel training for that group only).
- After retraining, global aggregation is performed again.

## Notes

- The code uses absolute paths internally; ensure you run `main.py` from the root directory (where this README is located).
- The script removes `/tmp/ray` at startup to avoid stale Ray processes. If you are running other Ray applications, be cautious.
