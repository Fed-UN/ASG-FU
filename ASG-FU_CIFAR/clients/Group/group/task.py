from sklearn.metrics import pairwise_distances
import glob
import re
import random
from flwr.common import ndarrays_to_parameters
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10
import os
import numpy as np
from torch.utils.data import DataLoader, Subset

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)#参数365分别代表输入通道数，输出通道数和卷积和大小
        # self.conv1 = nn.Conv2d(1, 6, 5)#替换为mnist后输入通道相应的修改为1个
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)#cifar-10数据集的输入图像尺寸上32*32
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)#修改为mnist对应的数据集中图像28*28的参数
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)#cifar-10数据集的输入图像尺寸上32*32
        # x = x.view(-1, 16 * 4 * 4)#修改为mnist对应的数据集中图像28*28的参数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def model_to_vector(model):
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def load_group_model_weights(model_path):
    net = Net()
    try:
        net.load_state_dict(torch.load(model_path, weights_only=True))
    except:
        print(f"Warning: Group model not found at {model_path}, using initial weights")
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in net.state_dict().items()])


def get_transforms():
    return Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10归一化参数
    ])


def load_data(partition_ids: list, num_partitions: int):
    base_path = "../../global-server/data_partitions"
    all_indices = []

    for pid in partition_ids:
        file_path = os.path.join(base_path, f"client_{pid}_all_indices.npy")
        indices = np.load(file_path, allow_pickle=True)
        all_indices.append(indices)

    all_indices = np.concatenate(all_indices).astype(int).tolist()
    transform = get_transforms()

    dataset = CIFAR10(
        root="../../data",
        train=True,
        download=False,
        transform=transform
    )

    np.random.shuffle(all_indices)
    split = int(0.8 * len(all_indices))
    train_indices = all_indices[:split]
    val_indices = all_indices[split:]

    trainloader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=32,
        shuffle=True,
    )

    valloader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=32,
    )

    return partition_ids, trainloader, valloader


def train(net, trainloader, epochs, device, client_id=None, round_num=0):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0014)
    net.train()
    running_loss = 0.0

    # 确保local_models目录存在
    os.makedirs("local_models", exist_ok=True)

    try:
        for _ in range(epochs):
            for batch in trainloader:
                images, labels = batch[0], batch[1]
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    except Exception as e:
        print(f"Training failed for client {client_id}: {str(e)}")
        running_loss = float('inf')

    avg_trainloss = running_loss / max(1, len(trainloader))

    # 强制保存模型
    if client_id is not None:
        model_path = os.path.join("local_models", f"client_{client_id}_round_{round_num}.pth")
        torch.save(net.state_dict(), model_path)

    return avg_trainloss


def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    try:
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch[0], batch[1]
                outputs = net(images.to(device))
                loss += criterion(outputs, labels.to(device)).item()
                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels.to(device)).sum().item()

        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
    except Exception as e:
        print(f"Testing failed: {str(e)}")
        loss = float('inf')
        accuracy = 0.0

    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def calculate_similarity(group_id, round_num, fixed_members):
    """计算组内客户端模型的相似度，使用固定成员列表"""
    model_dir = "local_models"
    client_models = {}

    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)

    for client_id in fixed_members:
        model_path = os.path.join(model_dir, f"client_{client_id}_round_{round_num}.pth")

        # 如果当前轮次模型不存在，尝试使用第0轮模型作为基础
        if not os.path.exists(model_path):
            base_model_path = os.path.join(model_dir, f"client_{client_id}_round_0.pth")
            if os.path.exists(base_model_path):
                model_path = base_model_path
                print(f"Using base model for client {client_id} in round {round_num}")

        if os.path.exists(model_path):
            try:
                net = Net()
                net.load_state_dict(torch.load(model_path, weights_only=True))
                model_vec = model_to_vector(net).numpy()
                client_models[client_id] = model_vec
            except Exception as e:
                print(f"Error loading model for client {client_id}: {str(e)}")
        else:
            print(f"Warning: Model not found for client {client_id} in round {round_num}")
            # 创建占位模型
            net = Net()
            model_vec = model_to_vector(net).numpy()
            client_models[client_id] = model_vec
            print(f"Created placeholder model for client {client_id}")

    if len(client_models) < 2:
        print(f"Not enough models ({len(client_models)}) to calculate similarity for group {group_id}")
        return None

    # 计算余弦相似度矩阵
    client_ids = sorted(client_models.keys())
    features = np.array([client_models[cid] for cid in client_ids])
    cosine_dist = pairwise_distances(features, metric='cosine')
    similarity_matrix = 1 - cosine_dist

    # 计算平均相似度
    avg_similarity = np.mean(similarity_matrix)

    # 构建相似度矩阵字典
    sim_matrix_dict = {}
    for i, cid_i in enumerate(client_ids):
        sim_matrix_dict[cid_i] = {}
        for j, cid_j in enumerate(client_ids):
            sim_matrix_dict[cid_i][cid_j] = float(similarity_matrix[i, j])

    # 计算每个客户端的平均相似度
    avg_similarities = {}
    for i, cid in enumerate(client_ids):
        # 排除自身相似度
        other_similarities = [sim for j, sim in enumerate(similarity_matrix[i]) if j != i]
        avg_sim = np.mean(other_similarities) if other_similarities else 0.0
        avg_similarities[cid] = float(avg_sim)

    # 正确选择聚合器和候选者
    # 1. 找到最高分
    max_score = max(avg_similarities.values())
    # 2. 所有具有最高分的客户端
    top_candidates = [cid for cid, score in avg_similarities.items() if score == max_score]
    # 3. 从最高分客户端中随机选择一个作为聚合器
    aggregator = random.choice(top_candidates) if top_candidates else None

    # 4. 找到第二高分（排除最高分）
    second_score = None
    second_candidates = []
    for score in sorted(set(avg_similarities.values()), reverse=True):
        if score < max_score:
            second_score = score
            break

    if second_score is not None:
        # 5. 所有具有第二高分的客户端
        second_candidates = [cid for cid, score in avg_similarities.items() if score == second_score]

    # 6. 确保候选者不包含聚合器
    if aggregator in second_candidates:
        second_candidates.remove(aggregator)

    # 构建结果字典
    result = {
        "members": client_ids,
        "aggregator": aggregator,
        "similarity": float(avg_similarity),
        "similarity_matrix": sim_matrix_dict,
        "selection_metrics": {
            "strategy": "max_average_similarity",
            "candidates": second_candidates,
            "scores": avg_similarities
        }
    }

    return result


def get_model_path_from_round(group_id, round_num):
    model_dir = "../../global-server/global_models"
    pattern = re.compile(rf"^{group_id}_round_{round_num}_accuracy_([0-9_]+)\.pth$")

    for filename in os.listdir(model_dir):
        if pattern.match(filename):
            return os.path.join(model_dir, filename)

    # 如果没有找到精确匹配，尝试找到最接近的模型
    all_models = glob.glob(os.path.join(model_dir, f"{group_id}_round_*.pth"))
    if not all_models:
        return None

    # 找到轮次最接近的模型
    best_match = None
    min_diff = float('inf')
    for model_path in all_models:
        match = re.search(rf'round_(\d+)', model_path)
        if match:
            model_round = int(match.group(1))
            diff = abs(model_round - round_num)
            if diff < min_diff:
                min_diff = diff
                best_match = model_path

    return best_match