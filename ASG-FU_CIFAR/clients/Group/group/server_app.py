import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from group.task import Net, load_group_model_weights, test, calculate_similarity, get_model_path_from_round
import datetime
import json
import logging
from collections import OrderedDict
from typing import Dict
import torch
from flwr.common import Context,Scalar, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import time

flwr_logger = logging.getLogger("flwr")
flwr_logger.setLevel(logging.ERROR)


class MaliciousEarlyStoppingStrategy(FedAvg):
    def __init__(self, shared_state, early_stopping_rounds, min_delta, max_delta, group_id, attack_config, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_state = shared_state
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.group_id = group_id
        self.attack_config = attack_config
        self._final_round = None

    def _apply_attack(self, server_round: int, parameters: NDArrays) -> NDArrays:
        if not self.attack_config["enable"]:
            return parameters
        if self.shared_state.get("rollback_required", False):
            return parameters
        if server_round < self.attack_config["start_round"] or server_round > self.attack_config["end_round"]:
            return parameters

        attack_type = self.attack_config["type"]
        strength = self.attack_config["strength"]

        attack_log = {
            "group_id": self.group_id,
            "round": server_round,
            "attack_type": attack_type,
            "strength": strength,
            "timestamp": datetime.datetime.now().isoformat()
        }

        attack_dir = os.path.abspath("../../attack_logs")
        os.makedirs(attack_dir, exist_ok=True)
        attack_file = os.path.join(attack_dir, f"{self.group_id}_round_{server_round}_attack.json")
        with open(attack_file, 'w') as f:
            json.dump(attack_log, f, indent=2)

        print(f"[MALICIOUS SERVER] Applying {attack_type} attack at round {server_round} (strength={strength})")

        if attack_type == "noise":
            return [p + np.random.normal(0, strength, p.shape) for p in parameters]
        elif attack_type == "negative_scaling":
            return [p * (1 - strength) for p in parameters]
        elif attack_type == "model_replacement":
            if server_round == self.attack_config["start_round"]:
                self.shared_state["backup_model"] = parameters
            if "malicious_model" not in self.shared_state:
                malicious_net = Net()
                for param in malicious_net.parameters():
                    param.data = torch.randn_like(param.data) * 0.1
                malicious_model = self.get_weights(malicious_net)
                self.shared_state["malicious_model"] = malicious_model
            return [
                (1 - strength) * backup + strength * malicious
                for backup, malicious in zip(
                    self.shared_state["backup_model"],
                    self.shared_state["malicious_model"]
                )
            ]
        elif attack_type == "dropout":
            return [
                p * (np.random.random(p.shape) > strength).astype(np.float32)
                for p in parameters
            ]

        return parameters

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if failures or aggregated[0] is None:
            return aggregated

        aggregated_params = parameters_to_ndarrays(aggregated[0])
        attacked_params = self._apply_attack(server_round, aggregated_params)
        return (ndarrays_to_parameters(attacked_params), aggregated[1])

    def get_weights(self, net):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def configure_fit(self, server_round, parameters, client_manager):
        # 检查是否触发了早停
        if self.shared_state.get("early_stop", False):
            #print(f"[Group {self.group_id}] Early stop triggered, skipping round {server_round}")
            return []

        # 检查是否需要回退模型
        if self.shared_state.get("rollback_required", False):
            rollback_round = self.shared_state["rollback_to_round"]
            model_path = get_model_path_from_round(self.group_id, rollback_round)
            if model_path and os.path.exists(model_path):
                net = Net()
                net.load_state_dict(torch.load(model_path, weights_only=True))
                parameters = self.get_weights(net)
                print(f"\n[Group {self.group_id} Rollback] Using rolled back model from round {rollback_round}")
                # 验证模型是否正确加载
                testloader = load_global_testset()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                loss, accuracy = test(net, testloader, device)
                print(f"[Group {self.group_id} Rollback] Rollback model accuracy: {accuracy:.4f}")
            # 重置回退标志
            self.shared_state["rollback_required"] = False

        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        # 检查是否触发了早停
        if self.shared_state.get("early_stop", False):
            #print(f"[Group {self.group_id}] Early stop triggered, skipping evaluation for round {server_round}")
            return []

        return super().configure_evaluate(server_round, parameters, client_manager)


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


def get_evaluate_fn(shared_state, early_stopping_rounds, min_delta, max_delta, group_id):
    testloader = load_global_testset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    if "accuracy_history" not in shared_state:
        shared_state["accuracy_history"] = {}
        shared_state["rollback_counter"] = 0
        shared_state["max_rollbacks"] = 1
        shared_state["fixed_members"] = []
        shared_state["early_stop"] = False

    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        if shared_state.get("early_stop", False):
            #print(f"[Group {group_id}] Early stop already triggered, skipping evaluation")
            return float('inf'), {"accuracy": 0.0}

        net = Net()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict)

        try:
            loss, accuracy = test(net, testloader, device)
        except Exception as e:
            print(f"[{group_id} Round {server_round}] Test failed: {str(e)}")
            loss = float('inf')
            accuracy = 0.0
        else:
            print(
                f"[Group {group_id} Round {server_round}] Accuracy: {accuracy:.4f} Total execution time: {time.time() - start_time:.3f}s")

        # 保存组模型
        model_dir = "../../global-server/global_models"
        os.makedirs(model_dir, exist_ok=True)
        formatted_accuracy = f"{accuracy:.4f}".replace('.', '_')
        filename = f"Group_{group_id}_round_{server_round}_accuracy_{formatted_accuracy}.pth"
        filepath = os.path.join(model_dir, filename)
        torch.save(net.state_dict(), filepath)

        # 记录日志
        log_dir = os.path.abspath(os.path.join(os.getcwd(), "../../training_logs"))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(os.path.join(log_dir, f"Group_{group_id}_log.csv"), "a") as f:
            if os.stat(os.path.join(log_dir, f"Group_{group_id}_log.csv")).st_size == 0:
                f.write("round,accuracy,time\n")
            f.write(f"{server_round},{accuracy:.4f},{current_time}\n")

        # 每3轮计算一次相似度
        if server_round % 3 == 0 and server_round != 0:
            similarity_result = calculate_similarity(
                group_id,
                server_round,
                shared_state["fixed_members"]
            )
            if similarity_result:
                sim_dir = os.path.abspath("similarity_logs")
                os.makedirs(sim_dir, exist_ok=True)
                sim_file = os.path.join(sim_dir, f"Group_{group_id}_round_{server_round}_similarity.json")
                with open(sim_file, 'w') as f:
                    json.dump(similarity_result, f, indent=2)
                print(f"[Group {group_id} Round {server_round}] Similarity calculated and saved")

        # 记录当前精度
        shared_state["accuracy_history"][server_round] = accuracy

        # 检查精度下降情况
        if shared_state.get("rollback_enabled", True) and len(shared_state["accuracy_history"]) >= 3:
            rounds = sorted(shared_state["accuracy_history"].keys())[-3:]
            accuracies = [shared_state["accuracy_history"][r] for r in rounds]

            # 检查是否连续两轮下降
            consecutive_decline = True
            if accuracies[1] >= (accuracies[0] - max_delta):
                consecutive_decline = False
            if accuracies[2] >= (accuracies[0] - max_delta):
                consecutive_decline = False

            if consecutive_decline:
                if shared_state["rollback_counter"] < shared_state["max_rollbacks"]:
                    rollback_round = rounds[0]
                    model_path = get_model_path_from_round(group_id, rollback_round)
                    if model_path:
                        shared_state["rollback_required"] = True
                        shared_state["rollback_to_round"] = rollback_round
                        shared_state["rollback_counter"] += 1

                        # 重置早停计数器
                        shared_state["best_loss"] = None
                        shared_state["no_improvement_count"] = 0
                        print(f"[Group {group_id} Rollback] Reset early stopping counters")

                        # 记录回退信息
                        rollback_info = {
                            "group_id": group_id,
                            "current_round": server_round,
                            "rollback_to_round": rollback_round,
                            "accuracies": {
                                str(rounds[0]): accuracies[0],
                                str(rounds[1]): accuracies[1],
                                str(rounds[2]): accuracies[2],
                            },
                            "rollback_count": shared_state["rollback_counter"],
                            "timestamp": datetime.datetime.now().isoformat()
                        }

                        monitor_dir = os.path.abspath("../../training_monitor")
                        os.makedirs(monitor_dir, exist_ok=True)
                        rollback_file = os.path.join(monitor_dir, f"{group_id}_rollback_round_{server_round}.json")
                        with open(rollback_file, 'w') as f:
                            json.dump(rollback_info, f, indent=2)

                        print(f"\n[Group {group_id} Rollback] Detected two consecutive accuracy declines")
                        print(f"[Group {group_id} Rollback] Rolling back to round {rollback_round}")
                        print(
                            f"[Group {group_id} Rollback] Rollback counter: {shared_state['rollback_counter']}/{shared_state['max_rollbacks']}")

        # 早停逻辑
        current_loss = loss
        best_loss = shared_state.get("best_loss")

        if best_loss is None:
            shared_state.update({
                "best_loss": current_loss,
                "no_improvement_count": 0
            })
        else:
            if current_loss > (best_loss - min_delta):
                shared_state["no_improvement_count"] += 1
                if shared_state["no_improvement_count"] >= early_stopping_rounds:
                    shared_state["early_stop"] = True
                    print(f"\n[Group {group_id} Early Stop] Training halted after round {server_round}")

                    # 创建停止标志
                    monitor_dir = os.path.abspath("../../training_monitor")
                    os.makedirs(monitor_dir, exist_ok=True)
                    flag_file = os.path.join(monitor_dir, f"Group_{group_id}_stopped.json")
                    with open(flag_file, 'w') as f:
                        json.dump({
                            "group_id": group_id,
                            "stop_round": server_round,
                            "timestamp": datetime.datetime.now().isoformat()
                        }, f)
            else:
                shared_state.update({
                    "best_loss": current_loss,
                    "no_improvement_count": 0
                })

        return loss, {"accuracy": accuracy}

    return evaluate_fn


def server_fn(context: Context):
    group_id = context.run_config.get('group_id', 0)
    model_path = os.path.join(
        "../../global-server/grouped_local_models",
        f"Group_{group_id}",
        f"Group_{group_id}.pth"
    )
    parameters = load_group_model_weights(model_path)
    num_rounds = 100
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    early_stopping_rounds = context.run_config.get("early-stopping-rounds", 3)
    min_delta = context.run_config.get("min-delta", 0.002)
    max_delta = context.run_config.get("max-delta", 0.01)

    # 加载分组配置获取固定成员列表
    cluster_file = "../../global-server/group_results/round1_clusters.json"
    with open(cluster_file, 'r') as f:
        clusters = json.load(f)

    group_name = f"Group_{group_id}"
    fixed_members = clusters.get(group_name, {}).get("members", [])

    attack_config = {
        "enable": context.run_config.get("attack-enable", False),
        "start_round": context.run_config.get("attack-start", 2),
        "end_round": context.run_config.get("attack-end", 3),
        "type": context.run_config.get("attack-type", "noise"),
        "strength": context.run_config.get("attack-strength", 0.05)
    }

    rollback_enabled = context.run_config.get("rollback-enabled", True)

    shared_state = {
        "early_stop": False,
        "best_loss": None,
        "no_improvement_count": 0,
        "accuracy_history": {},
        "rollback_required": False,
        "rollback_counter": 0,
        "max_rollbacks": 3,
        "rollback_enabled": rollback_enabled,
        "fixed_members": fixed_members
    }

    def fit_config_fn(server_round: int):
        return {"server_round": server_round}

    strategy = MaliciousEarlyStoppingStrategy(
        shared_state=shared_state,
        early_stopping_rounds=early_stopping_rounds,
        min_delta=min_delta,
        max_delta=max_delta,
        group_id=group_id,
        attack_config=attack_config,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_available_clients=1,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(shared_state, early_stopping_rounds, min_delta, max_delta, group_id),
        on_fit_config_fn=fit_config_fn
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )


app = ServerApp(server_fn=server_fn)