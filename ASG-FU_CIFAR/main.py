import datetime
import os
import json
import shutil
import subprocess
import random
import time
from collections import defaultdict

import numpy as np
import torch
from subprocess import Popen
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import glob
import re
from scipy.spatial.distance import cdist
import subprocess

def cleanup_previous_runs():
    subprocess.run(["ray", "stop", "--force"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    client_groups = glob.glob("clients/Group_*")
    for path in client_groups:
        if os.path.isdir(path):
            shutil.rmtree(path)
    cleanup_dirs = [
        "global-server/data_partitions",
        "global-server/global_models",
        "global-server/group_results",
        "global-server/grouped_local_models",
        "global-server/local_models",
        "training_logs"
    ]
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    print("Old files removed")

    # 新增：删除 /tmp/ray 文件夹
    ray_tmp_path = "/tmp/ray"
    if os.path.exists(ray_tmp_path):
        shutil.rmtree(ray_tmp_path)
        #print(f"Removed {ray_tmp_path}")


def evaluate_model_accuracy(model):
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10归一化参数
    ])
    testset = CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transform
    )
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
    return correct / total


class GroupNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)


def model_to_vector(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def run_global_server():
    global_server_dir = os.path.abspath("global-server")
    subprocess.check_call(["flwr", "run", "."], cwd=global_server_dir)
    print("\nFirst global aggregation round completed.")


def perform_grouping():
    global_server_dir = os.path.abspath("global-server")
    artifacts = {
        "local_models": os.path.join(global_server_dir, "local_models"),
        "group_results": os.path.join(global_server_dir, "group_results"),
        "grouped_models": os.path.join(global_server_dir, "grouped_local_models")
    }

    # Load client models
    client_models = []
    client_ids = []
    for model_path in glob.glob(os.path.join(artifacts["local_models"], "client_*_round_1.pth")):
        client_id = int(os.path.basename(model_path).split('_')[1])
        net = GroupNet()
        net.load_state_dict(torch.load(model_path, weights_only=True))
        client_models.append(net)
        client_ids.append(client_id)

    n_clients = len(client_ids)
    if n_clients == 0:
        return

    # Feature extraction
    features = torch.stack([model_to_vector(m) for m in client_models]).numpy()

    # Determine optimal cluster count
    k = determine_optimal_clusters(features, n_clients)

    # Initial K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)

    # Intelligent rebalancing
    labels = intelligent_rebalance(kmeans.labels_, features, n_clients)

    # Build final groups
    groups = defaultdict(list)
    for cid, label in zip(client_ids, labels):
        groups[label].append(cid)

    # Save grouping configuration
    save_group_config(dict(groups), artifacts["group_results"], features, client_ids)
    organize_group_models(groups, artifacts)
    print(f"\nGrouping completed: {len(groups)} groups, sizes: {[len(v) for v in groups.values()]}")


def determine_optimal_clusters(features, n_clients):
    """Determine optimal number of clusters with combined metrics"""
    # if n_clients >= 1:
    #     return int(np.round(np.sqrt(n_clients)))
    if n_clients <= 1:
        return 1

    # Base candidate k range around square root
    base_k = int(np.round(np.sqrt(n_clients)))
    k_min = max(2, base_k - 5)
    k_max = min(base_k + 5, n_clients - 1)
    k_candidates = range(k_min, k_max + 1)
    k_penalty_weights = {
        k: np.exp(-0.5 * ((k - base_k) / (base_k * 0.5)) ** 2)
        for k in k_candidates
    }
    best_k = base_k
    best_score = -np.inf
    score_records = []

    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(features)
        labels = kmeans.labels_

        # Calculate metrics
        sil_score = silhouette_score(features, labels, metric='cosine') if k > 1 else -1
        _, counts = np.unique(labels, return_counts=True)
        size_var = np.var(counts) / (n_clients ** 2) if len(counts) > 0 else 0

        # Normalize scores
        sil_norm = (sil_score - (-1)) / (1 - (-1))
        var_norm = 1 - size_var

        combined_score = (
                sil_norm +
                var_norm +
                k_penalty_weights[k]
        )

        score_records.append((k, sil_score, size_var, combined_score))

        if combined_score > best_score:
            best_score = combined_score
            best_k = k

    # Print debugging info
    # print(f"\n[Optimal K] Base k={base_k}, candidates: {list(k_candidates)}")
    # print("k | Silhouette | Size Var | Combined")
    # for r in score_records:

    #     print(f"{r[0]:2} | {r[1]:.3f} | {r[2]:.4f} | {r[3]:.3f}")
    # print(f"Optimal k={best_k} (score={best_score:.3f})")
    return best_k


def intelligent_rebalance(initial_labels, features, n_clients):
    """基于最小客户端数量的智能再平衡"""
    k = len(np.unique(initial_labels))
    target_size = n_clients // k
    max_size = target_size + 1 if n_clients % k != 0 else target_size
    min_size = target_size  # 核心成员数量

    # 阶段1：提取核心成员
    groups = [[] for _ in range(k)]
    for idx, lbl in enumerate(initial_labels):
        groups[lbl].append(idx)

    core_groups = []
    overflow = []
    for gid, members in enumerate(groups):
        if len(members) <= min_size:
            core_groups.append(members.copy())
            continue

        # 计算组内特征中心
        group_features = features[members]
        centroid = np.mean(group_features, axis=0)

        # 按距中心距离排序
        distances = cdist(group_features, [centroid], metric='cosine').flatten()
        sorted_idx = np.argsort(distances)[:min_size]  # 取最近的min_size个

        core_members = [members[i] for i in sorted_idx]
        core_groups.append(core_members)
        overflow.extend([members[i] for i in range(len(members)) if i not in sorted_idx])

    # 阶段2：动态分配溢出成员
    def find_optimal_group(client_id):
        client_feat = features[client_id]
        candidates = []

        for gid, group in enumerate(core_groups):
            current_size = len(group)
            if current_size >= max_size:
                continue

            # 计算组当前特征中心
            group_feats = features[group]
            group_center = np.mean(group_feats, axis=0)

            # 计算相似度（余弦相似度）
            similarity = 1 - cdist([client_feat], [group_center], metric='cosine').item()

            # 优先级标准：1.当前成员数最少 2.相似度最高
            candidates.append((-current_size, similarity, gid))

        if not candidates:
            return -1

        # 按优先级排序（成员数升序 -> 相似度降序）
        candidates.sort(reverse=True)
        return candidates[0][2]

    # 随机顺序处理溢出成员
    np.random.shuffle(overflow)
    for client_id in overflow:
        target_gid = find_optimal_group(client_id)
        if target_gid != -1:
            core_groups[target_gid].append(client_id)
        else:
            # 所有组已满时创建新组（理论上不应该发生）
            core_groups.append([client_id])

    # 阶段3：递归处理超限分组
    new_overflow = []
    for gid, group in enumerate(core_groups):
        if len(group) > max_size:
            # 重新计算中心并保留核心成员
            group_feats = features[group]
            centroid = np.mean(group_feats, axis=0)
            distances = cdist(group_feats, [centroid], metric='cosine').flatten()
            sorted_idx = np.argsort(distances)[:max_size]

            new_core = [group[i] for i in sorted_idx]
            new_overflow.extend([group[i] for i in range(len(group)) if i not in sorted_idx])
            core_groups[gid] = new_core

    if new_overflow:
        # 构造虚拟初始标签进行递归处理
        virtual_labels = np.zeros(n_clients, dtype=int)
        for gid, group in enumerate(core_groups):
            for idx in group:
                virtual_labels[idx] = gid
        return intelligent_rebalance(virtual_labels, features, n_clients)

    # 最终容量验证
    final_sizes = [len(g) for g in core_groups]
    if max(final_sizes) > max_size or min(final_sizes) < min_size:
        return intelligent_rebalance(initial_labels, features, n_clients)

    # 生成最终标签
    final_labels = np.zeros(n_clients, dtype=int)
    for gid, group in enumerate(core_groups):
        for idx in group:
            final_labels[idx] = gid

    return final_labels


def save_group_config(groups, output_dir, features, client_ids):
    config = {}
    for lbl, members in groups.items():
        group_name = f"Group_{lbl}"
        member_indices = [client_ids.index(cid) for cid in members]
        group_features = features[member_indices]

        cosine_dist = pairwise_distances(group_features, metric='cosine')
        similarity_matrix = 1 - cosine_dist

        n_members = len(members)
        if n_members > 1:
            row_sums = np.sum(similarity_matrix, axis=1) - 1
            avg_similarities = row_sums / (n_members - 1)
        else:
            avg_similarities = np.array([1.0])

        max_score = np.max(avg_similarities)
        candidates = [members[i] for i, score in enumerate(avg_similarities) if score == max_score]

        sim_matrix_dict = {
            cid_i: {cid_j: float(similarity_matrix[i][j])
                    for j, cid_j in enumerate(members)}
            for i, cid_i in enumerate(members)
        }

        aggregator = int(random.choice(candidates)) if candidates else int(members[0])

        config[group_name] = {
            "members": members,
            "aggregator": aggregator,
            "similarity": float(np.mean(similarity_matrix)),
            "similarity_matrix": sim_matrix_dict,
            "selection_metrics": {
                "strategy": "max_average_similarity",
                "candidates": candidates,
                "scores": {cid: float(avg_similarities[i])
                           for i, cid in enumerate(members)}
            }
        }

    config_path = os.path.join(output_dir, "round1_clusters.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    time.sleep(1)


def organize_group_models(groups, artifacts):
    for lbl, members in groups.items():
        group_dir = os.path.join(artifacts["grouped_models"], f"Group_{lbl}")
        os.makedirs(group_dir, exist_ok=True)
        for cid in members:
            src = os.path.join(artifacts["local_models"], f"client_{cid}_round_1.pth")
            if os.path.exists(src):
                shutil.copy2(src, group_dir)
        aggregate_models(group_dir)
        print(f"Group {lbl} initialized with {len(members)} client models")


def aggregate_models(group_dir):
    model_files = glob.glob(os.path.join(group_dir, "client_*.pth"))
    if not model_files:
        return
    models, accuracies = [], []
    for f in model_files:
        model = GroupNet()
        model.load_state_dict(torch.load(f, weights_only=True))
        acc = evaluate_model_accuracy(model)
        models.append(model)
        accuracies.append(acc)
    weights = torch.softmax(torch.tensor(accuracies), dim=0)
    state_dict = {}
    for key in models[0].state_dict():
        params = [m.state_dict()[key].float() * w for m, w in zip(models, weights)]
        state_dict[key] = sum(params)
    aggregated = GroupNet()
    aggregated.load_state_dict(state_dict)
    save_path = os.path.join(group_dir, f"{os.path.basename(group_dir)}.pth")
    torch.save(aggregated.state_dict(), save_path)


def wait_all_groups_ready(group_ids):
    """Wait for all groups to complete training"""
    monitor_dir = os.path.abspath("training_monitor")
    start_time = time.time()
    completed = set()

    print(f"Current batch groups: {group_ids}")
    #subprocess.check_call([
        #"ray", "start", "--head",
        #"--dashboard-port=8285", 
        #"--min-worker-port=20000",
        #"--max-worker-port=29999"
    #])
    while True:
        new_completed = set()
        for gid in group_ids:
            flag_file = os.path.join(monitor_dir, f"Group_{gid}_stopped.json")
            if gid not in completed and os.path.exists(flag_file):
                completed.add(gid)
                new_completed.add(gid)

        if new_completed:
            for gid in sorted(new_completed):
                print(f"Group_{gid} training completed")

        if len(completed) == len(group_ids):
            total_time = time.time() - start_time
            print(f"Batch completed! Total time: {total_time:.1f}s\n")
            return

        time.sleep(1)


def load_best_group_model(group_id):
    model_dir = os.path.abspath(f"global-server/global_models")
    model_files = glob.glob(os.path.join(model_dir, f"Group_{group_id}_round_*.pth"))
    model_files.sort(
        key=lambda f: int(re.search(r'_round_(\d+)', f).group(1)) if re.search(r'_round_(\d+)', f) else 0,
        reverse=True
    )
    model = GroupNet()
    model.load_state_dict(torch.load(model_files[0], weights_only=True))
    print(f"[Loaded] Group_{group_id} latest model: {os.path.basename(model_files[0])}")
    return model


def global_aggregation(num_groups, processes=None):
    """Global model aggregation with dynamic weighting"""
    transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    testset = CIFAR10(root="data", train=False, transform=transform, download=False)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    group_models, group_accs, group_sizes = [], [], []
    print(f"\n[Evaluating] Collecting {num_groups} group models:")

    for gid in range(num_groups):
        model = load_best_group_model(gid)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        group_models.append(model)
        group_accs.append(acc)
        group_sizes.append(5000)  # Assume 5000 samples per group
        print(f"Group_{gid}, Accuracy: {acc:.4f}")

    # Weight calculation
    accuracy_weights = torch.softmax(torch.tensor(group_accs) * 5, dim=0)
    data_weights = torch.tensor(group_sizes) / sum(group_sizes)
    combined_weights = 0.7 * accuracy_weights + 0.3 * data_weights

    print("\n[Weight Allocation]:")
    for gid, (a, d, w) in enumerate(zip(accuracy_weights, data_weights, combined_weights)):
        print(f"Group_{gid} | Acc Weight: {a:.4f} | Data Weight: {d:.4f} | Combined: {w:.4f}")

    # Model aggregation
    final_model = GroupNet()
    final_state = {}
    for key in group_models[0].state_dict():
        params = [m.state_dict()[key].float() * w for m, w in zip(group_models, combined_weights)]
        final_state[key] = sum(params)
    final_model.load_state_dict(final_state)

    # Final evaluation
    final_correct, final_total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = final_model(images)
            _, predicted = torch.max(outputs.data, 1)
            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()
    final_acc = final_correct / final_total
    print(f"Final accuracy: {final_acc:.4f}")

    # Save logs
    log_dir = os.path.abspath(os.path.join(os.getcwd(), "training_logs"))
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = {
        'accuracy': f"{final_acc:.4f}",
        'time': current_time
    }
    log_file = os.path.join(log_dir, 'agg_log.csv')
    write_header = not os.path.exists(log_file) or os.stat(log_file).st_size == 0
    with open(log_file, 'a') as f:
        if write_header:
            f.write("accuracy,time\n")
        f.write(f"{log_entry['accuracy']},{log_entry['time']}\n")

    # Terminate processes
    if processes:
        for proc in processes:
            proc.terminate()
        print("\nTerminated all group server processes")


def copy_group_templates(num_groups):
    """Create client group directories from template"""
    template = os.path.abspath("clients/group")
    for gid in range(num_groups):
        dest = os.path.abspath(f"clients/Group_{gid}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(template, dest)
        print(f"Created client group: Group_{gid}")


def modify_group_ids(num_groups):
    """Update group IDs in configuration files"""
    for gid in range(num_groups):
        toml_path = os.path.abspath(f"clients/Group_{gid}/pyproject.toml")
        with open(toml_path, 'r+') as f:
            content = [line if not line.startswith('group_id') else f'group_id = {gid}\n'
                       for line in f.readlines()]
            f.seek(0)
            f.writelines(content)
            f.truncate()


def run_all_groups_parallel(group_ids):
    """Start all group servers simultaneously"""
    processes = []
    for gid in group_ids:
        group_dir = os.path.abspath(f"clients/Group_{gid}")
        proc = Popen(["flwr", "run", "."], cwd=group_dir)
        processes.append(proc)
        print(f"Started Group_{gid} (PID: {proc.pid})")

    # Wait for all groups to complete
    wait_all_groups_ready(group_ids)
    cleanup_monitor_files(group_ids)
    return processes


def cleanup_monitor_files(group_ids):
    """Clean up monitoring files for specified groups"""
    monitor_dir = os.path.abspath("training_monitor")
    for gid in group_ids:
        flag_file = os.path.join(monitor_dir, f"Group_{gid}_stopped.json")
        if os.path.exists(flag_file):
            os.remove(flag_file)


def cleanup_group_models(group_ids):
    """Remove old model files for specified groups"""
    model_dir = os.path.abspath("global-server/global_models")
    for gid in group_ids:
        pattern = os.path.join(model_dir, f"Group_{gid}_round_*.pth")
        for f in glob.glob(pattern):
            os.remove(f)
            print(f"Removed old model: {os.path.basename(f)}")


def unlearning(ratio):
    """Perform unlearning operation on selected groups with model re-aggregation"""
    cluster_path = "global-server/group_results/round1_clusters.json"
    with open(cluster_path, 'r') as f:
        cluster_data = json.load(f)

    num_groups = len(cluster_data)
    selected_num = int(num_groups * ratio)
    selected_num = 1
    if selected_num <= 0:
        return []
    selected_group_ids = random.sample(range(num_groups), selected_num)

    # Clean old models
    cleanup_group_models(selected_group_ids)
    
    groups_to_reaggregate = []

    for gid in selected_group_ids:
        group_name = f"Group_{gid}"
        members = cluster_data[group_name]["members"]
        if len(members) > 1:
            # Remove a random client
            removed_client = members.pop(random.randint(0, len(members) - 1))
            print(f"[Unlearning] Group {group_name} removed client {removed_client}, remaining: {members}")
            cluster_data[group_name]["members"] = members
            
            # 添加到需要重新聚合的组列表
            groups_to_reaggregate.append((gid, members))
    
    # 重新聚合剩余客户端的模型
    for gid, remaining_members in groups_to_reaggregate:
        reaggragate_group_models(gid, remaining_members)
    
    # Save modified configuration
    cluster_unlearning_path = "global-server/group_results/round1_unlearning_clusters.json"
    with open(cluster_unlearning_path, 'w') as f:
        json.dump(cluster_data, f, indent=2)
    
    return [gid for gid, _ in groups_to_reaggregate]

def reaggragate_group_models(group_id, remaining_members):
    """重新聚合组内剩余客户的模型"""
    artifacts = {
        "local_models": os.path.abspath("global-server/local_models"),
        "grouped_models": os.path.abspath("global-server/grouped_local_models")
    }
    
    # 1. 创建组目录
    group_dir = os.path.join(artifacts["grouped_models"], f"Group_{group_id}")
    os.makedirs(group_dir, exist_ok=True)
    
    # 2. 复制剩余成员的模型
    for cid in remaining_members:
        src = os.path.join(artifacts["local_models"], f"client_{cid}_round_1.pth")
        if os.path.exists(src):
            shutil.copy2(src, group_dir)
    
    # 3. 聚合剩余成员的模型
    model_files = glob.glob(os.path.join(group_dir, "client_*.pth"))
    models = []
    for f in model_files:
        model = GroupNet()
        model.load_state_dict(torch.load(f, weights_only=True))
        models.append(model)
    
    # 简单平均聚合
    state_dict = {}
    for key in models[0].state_dict():
        params = [m.state_dict()[key].float() for m in models]
        state_dict[key] = sum(params) / len(models)
    
    # 4. 保存新聚合的模型
    aggregated = GroupNet()
    aggregated.load_state_dict(state_dict)
    save_path = os.path.join(group_dir, f"Group_{group_id}.pth")
    torch.save(aggregated.state_dict(), save_path)
    print(f"[Unlearning] Group {group_id} re-aggregated with {len(remaining_members)} clients")



if __name__ == "__main__":
    start_time = time.time()
    cleanup_previous_runs()
    monitor_dir = os.path.abspath("training_monitor")
    shutil.rmtree(monitor_dir, ignore_errors=True)

    # Initial training phase
    run_global_server()
    perform_grouping()

    with open("global-server/group_results/round1_clusters.json") as f:
        cluster_data = json.load(f)
        num_groups = len(cluster_data)

    copy_group_templates(num_groups)
    modify_group_ids(num_groups)

    # First training round with all groups
    all_groups = list(range(num_groups))
    processes = run_all_groups_parallel(all_groups)
    global_aggregation(num_groups, processes)

    # # Unlearning phase
    unlearn_time = time.time()
    unlearn_ratio = 0.1
    selected_groups = unlearning(unlearn_ratio)
    cleanup_monitor_files(selected_groups)
    processes = run_all_groups_parallel(selected_groups)
    global_aggregation(num_groups, processes)
    print(f"Unlearning time: {time.time() - unlearn_time:.2f}s")

    print(f"Total execution time: {time.time() - start_time:.2f}s")