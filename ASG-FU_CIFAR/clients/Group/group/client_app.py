from group.task import Net, get_weights, load_data, set_weights, test, train
import time
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
import json
import os

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, global_testloader, local_epochs, client_id, group_id, fixed_members):
        self.net = net
        self.trainloader = trainloader
        self.valloader = global_testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.client_id = client_id
        self.group_id = group_id
        self.fixed_members = fixed_members
        # 确保模型目录存在
        os.makedirs("local_models", exist_ok=True)

        # 初始化模型文件
        self.initialize_model_files()

    def initialize_model_files(self):
        """确保所有固定成员在第0轮都有初始模型文件"""
        for cid in self.fixed_members:
            model_path = os.path.join("local_models", f"client_{cid}_round_0.pth")
            if not os.path.exists(model_path):
                # 创建初始模型
                net = Net()
                torch.save(net.state_dict(), model_path)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        server_round = config.get("server_round", 1)
        start_time = time.time()

        # 确保所有固定成员都有模型文件
        self.ensure_all_members_have_model(server_round, parameters)

        # 当前客户端训练
        try:
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
                self.client_id,
                server_round
            )
        except Exception as e:
            print(f"Client {self.client_id} training failed: {str(e)}")
            train_loss = float('inf')

        train_time = time.time() - start_time
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": train_loss}

    def ensure_all_members_have_model(self, server_round, parameters):
        """确保所有固定成员都有当前轮次的模型文件"""
        for cid in self.fixed_members:
            model_path = os.path.join("local_models", f"client_{cid}_round_{server_round}.pth")
            if not os.path.exists(model_path):
                # 如果当前客户端，使用当前参数
                if cid == self.client_id:
                    torch.save(self.net.state_dict(), model_path)
                else:
                    # 对于其他客户端，创建占位模型
                    net = Net()
                    set_weights(net, parameters)
                    torch.save(net.state_dict(), model_path)

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        try:
            loss, accuracy = test(self.net, self.valloader, self.device)
        except Exception as e:
            print(f"Client {self.client_id} evaluation failed: {str(e)}")
            loss = float('inf')
            accuracy = 0.0
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def load_global_testset():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10归一化参数
    ])
    testset = CIFAR10(
        root="../../data",
        train=False,
        download=False,
        transform=transform
    )
    return DataLoader(testset, batch_size=32, shuffle=False)


def client_fn(context: Context):
    group_id = context.run_config.get('group_id', 0)
    cluster_file = "../../global-server/group_results/round1_clusters.json"

    with open(cluster_file, 'r') as f:
        clusters = json.load(f)

    group_name = f"Group_{group_id}"
    group_info = clusters.get(group_name, {})
    partition_ids = group_info.get("members", [])
    fixed_members = partition_ids.copy()

    client_idx = context.node_id % len(partition_ids)
    client_id = partition_ids[client_idx]

    _, trainloader, valloader = load_data(partition_ids, len(partition_ids))
    local_epochs = context.run_config.get("local-epochs", 1)
    net = Net()
    global_testloader = load_global_testset()

    return FlowerClient(net, trainloader, global_testloader, local_epochs,
                        client_id, group_id, fixed_members).to_client()


app = ClientApp(client_fn=client_fn)