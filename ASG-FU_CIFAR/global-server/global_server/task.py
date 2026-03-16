"""CIFAR-10 Federated Learning with Dirichlet Non-IID Partitioning"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10
import os
import random

# 全局配置
RANDOM_SEED = 42
DIRICHLET_ALPHA = 0.5  # Dirichlet分布参数，值越小数据异构性越强

# 确保可复现性
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 模型定义（适配CIFAR-10）
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

# 数据预处理
def get_transforms():
    return Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

# 核心改进：Dirichlet划分+索引不重叠保证
def dirichlet_split(dataset, num_clients, alpha=DIRICHLET_ALPHA):
    """使用Dirichlet分布划分数据集，确保客户端索引不重叠"""
    labels = np.array([target for _, target in dataset])
    num_classes = len(np.unique(labels))
    
    # 按类别组织索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    # 对每个类别进行Dirichlet划分
    for c_indices in class_indices:
        np.random.shuffle(c_indices)
        
        # 生成类别分配比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(c_indices)).astype(int)
        
        # 调整分配数量确保总和正确
        diff = len(c_indices) - proportions.sum()
        if diff != 0:
            proportions[np.argmax(proportions)] += diff
        
        # 分配索引到客户端
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(c_indices[start_idx:end_idx])
            start_idx = end_idx
    
    return client_indices

# 全局数据缓存
GLOBAL_TRAIN_DATASET = None
GLOBAL_SPLIT = None

def load_data(partition_id: int, num_partitions: int):
    """加载客户端数据，包含Dirichlet划分和本地训练/验证分割"""
    global GLOBAL_TRAIN_DATASET, GLOBAL_SPLIT
    
    # 全局数据集初始化
    if GLOBAL_TRAIN_DATASET is None:
        transform = get_transforms()
        GLOBAL_TRAIN_DATASET = CIFAR10(
            root="../data", 
            train=True, 
            download=True,
            transform=transform
        )
    
    # 全局划分初始化
    if GLOBAL_SPLIT is None:
        split_file = os.path.join("data_partitions", f"dirichlet_split_{num_partitions}.npy")
        
        if os.path.exists(split_file):
            GLOBAL_SPLIT = np.load(split_file, allow_pickle=True)
        else:
            os.makedirs("data_partitions", exist_ok=True)
            GLOBAL_SPLIT = dirichlet_split(GLOBAL_TRAIN_DATASET, num_partitions)
            np.save(split_file, np.array(GLOBAL_SPLIT, dtype=object), allow_pickle=True)
    
    # 获取当前客户端数据
    client_indices = GLOBAL_SPLIT[partition_id]
    train_dataset = Subset(GLOBAL_TRAIN_DATASET, client_indices)
    
    # 本地训练/验证分割 (80/20)
    num_train = int(0.8 * len(train_dataset))
    num_val = len(train_dataset) - num_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # 保存索引文件
    os.makedirs("data_partitions", exist_ok=True)
    
    # 客户端内部索引处理
    train_local_indices = [train_dataset.indices[i] for i in train_subset.indices]
    val_local_indices = [train_dataset.indices[i] for i in val_subset.indices]
    
    np.save(os.path.join("data_partitions", f"client_{partition_id}_all_indices.npy"), client_indices)
    np.save(os.path.join("data_partitions", f"client_{partition_id}_train_indices.npy"), np.array(train_local_indices))
    np.save(os.path.join("data_partitions", f"client_{partition_id}_val_indices.npy"), np.array(val_local_indices))
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32)
    
    return train_loader, val_loader

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.004)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch[0]
            labels = batch[1]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)