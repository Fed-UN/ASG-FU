"""Global-Server: A Flower / PyTorch app."""
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from global_server.task import Net, get_weights, load_data, set_weights, test, train
import os

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, CIFAR10
from torchvision.transforms import Normalize, ToTensor, Compose


# 在client_app.py、server_app.py和task.py中统一修改数据加载部分
def load_global_testset():
    transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10归一化参数
    ])
    testset = CIFAR10(root="../data", train=False, download=False, transform=transform)
    return DataLoader(testset, batch_size=32, shuffle=False)

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, partition_id):  # 添加partition_id参数
        self.net = net
        self.trainloader = trainloader
        self.valloader =load_global_testset()
        self.local_epochs = local_epochs
        self.partition_id = partition_id  # 新增属性
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        server_round = config.get("server_round", 1)  # 获取当前轮次
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        # 保存第一轮的模型
        if server_round == 1:
            os.makedirs("local_models", exist_ok=True)
            model_path = os.path.join("local_models", f"client_{self.partition_id}_round_1.pth")
            torch.save(self.net.state_dict(), model_path)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(net, trainloader, valloader, local_epochs, partition_id).to_client()

app = ClientApp(
    client_fn,
)