"""Global-Server: A Flower / PyTorch app."""
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from global_server.task import Net, get_weights, test  # 确保导入test函数
import logging
import datetime
import os
from collections import OrderedDict
import torch
from setuptools.command.build_ext import if_dl
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)

def load_global_testset():
    """加载全局测试集"""
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))  # 与客户端相同的预处理
    ])
    testset = FashionMNIST(root="../data", train=False, transform=transform)
    return DataLoader(testset, batch_size=32, shuffle=False)

def get_evaluate_fn():
    """创建评估函数用于记录日志"""
    testloader = load_global_testset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate_fn(server_round: int, parameters, config):
        # 加载全局模型参数
        net = Net()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict)

        # 在全局测试集上评估
        loss, accuracy = test(net, testloader, device)

        # 创建日志目录并写入结果
        log_dir = os.path.abspath(os.path.join(os.getcwd(), "../training_logs"))
        os.makedirs(log_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # 写入CSV文件
        log_file = os.path.join(log_dir, "global_training_log.csv")
        if server_round == 0:
            with open(log_file, "a") as f:
                if os.path.getsize(log_file) == 0:  # 添加表头
                    f.write("round,accuracy,time\n")
                f.write(f"{server_round},{accuracy:.4f},{current_time}\n")
        return loss, {"accuracy": accuracy}
    return evaluate_fn

def server_fn(context: Context):
    """配置服务器策略并集成评估函数"""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # 初始化模型参数
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # 配置联邦平均策略
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,  # 每次评估所有可用客户端
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        evaluate_fn=get_evaluate_fn()  # 集成评估函数
    )

    # 设置训练轮次
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)