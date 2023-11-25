import random
import json

import torch
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from torchvision import models
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm


def get_dataset(dir, name, download=False):
    """
    获得训练和测试使用的数据集的函数
    :param dir: 数据集下载或是使用的目录
    :param name: mnist or cifar 图像分类数据集
    :param download: 是否要下载数据集
    :return: train_dataset, test_dataset
    """
    if name == 'mnist':
        train_dataset = datasets.MNIST(dir, train=True, download=download,
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                        (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(dir, train=True, download=download, transform=transform_train)
        test_dataset = datasets.CIFAR10(dir, train=False, download=download, transform=transform_test)

    return train_dataset, test_dataset


class Server(object):
    """横向联邦学习使用的服务器"""

    def __init__(self, conf, test_dataset):
        """
        初始化服务器配置
        :param conf: 基本配置
        :param test_dataset: 验证使用的DataLoader对象
        """
        self.conf = conf
        # 定义全局模型
        self.global_model = models.get_model(self.conf['model_name'])
        # 获得验证使用的DataLoader对象
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.conf['batch_size'], shuffle=True)

    def model_aggregate(self, weight_accumulator):
        """
        模型聚合函数
        :param weight_accumulator: 存储了每一个客户端的上传参数的变化值
        :return: None
        """
        for name, data in self.global_model.state_dict().items():
            # 获得每一层需要更新的量 lambda为更新的权重
            update_per_layer = weight_accumulator[name] * self.conf['lambda']
            # 遍历全局模型的每一层并更新参数 确保类型相同
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer)
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        # 进行全局模型的评估
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for idx, data in enumerate(self.test_loader):
            # 获得验证的数据
            sample, label = data
            sample = sample.to(self.device)
            label = label.to(self.device)
            # 增加数据集的量
            dataset_size += data.size()[0]
            # 查看训练设备
            if torch.cuda.is_available():
                sample = sample.cuda()
                label = label.cuda()
            # 数据的正向传播
            output = self.global_model(sample)
            # 使用交叉熵损失函数并进行聚合
            total_loss += F.cross_entropy(output, label, reduction='sum').item()
            # 获取最大的对数概率的索引值
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
        # 计算全局模型的准确率
        acc = 100.0 * (float(correct) / float(dataset_size))
        # 计算平均损失值
        total_l = total_loss / dataset_size
        return acc, total_l


class Client(object):
    """横向联邦学习的客户端"""

    def __init__(self, conf, model, train_dataset, idx=1):
        """

        :param conf: 配置文件
        :param model: 客户端本地模型
        :param train_dataset: 客户端ID
        :param idx: 客户端本地的训练数据集
        """
        self.conf = conf
        self.local_model = model
        self.client_id = idx
        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        # 训练数据集的分配
        indices = all_range[idx * data_len: (idx + 1) * data_len]
        # 获取训练使用的DataLoader对象
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=conf['batch_size'],
                                                        sampler=SubsetRandomSampler(indices))
        # 进度条的长度
        self.length = int(data_len / conf['batch_size'])

    def local_train(self, model):
        # 本地模型的训练函数
        for name, param in model.state_dict().items():
            # 客户端首先使用服务器端下发的全局模型覆盖本地模型
            self.local_model.state_dict()[name].copy_(param.clone())
        # 定义最优化函数器 用于本地训练
        optimizer = torch.optim.SGD(self.local_model.parameters(),
                                    lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        # 本地模型训练
        self.local_model.train()
        for epoch in range(self.conf['local_epochs']):
            with tqdm(total=self.length, desc=f'Epoch {epoch + 1}') as pbar:
                for batch_idx, data in enumerate(self.train_loader):
                    sample, label = data
                    # 检查本地设备
                    if torch.cuda.is_available():
                        sample = sample.cuda()
                        label = label.cuda()
                    # 清空原本的梯度
                    optimizer.zero_grad()
                    # 网络正向传播
                    output = self.local_model(sample)
                    loss = F.cross_entropy(output, label)
                    # 误差反向传播
                    loss.backward()
                    # 模型的更新
                    optimizer.step()
                pbar.set_postfix({'Loss': loss})
        # 将更新后的模型参数放入一个字典中
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff


conf = {
    "model_name": "resnet18",
    "no_models": 10,
    "type": "cifar",
    "global_epochs": 20,
    "local_epochs": 3,
    "k": 6,
    "batch_size": 32,
    "lr": 0.001,
    "momentum": 0.001,
    "lambda": 0.1
}

train_dataset, test_dataset = get_dataset("./data/", conf["type"], download=True)
server = Server(conf, test_dataset)

# 创建客户端的列表
clients = []
for c in range(conf['no_models']):
    clients.append(Client(conf, server.global_model, train_dataset, c))

for e in range(conf['global_epochs']):
    # 采样K个客户端参与本轮联邦训练
    candidates = random.sample(clients, conf['k'])

    weight_accumulator = {}
    for name, params in server.global_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)

    for c in candidates:
        diff = c.local_train(server.global_model)
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name].add_(diff[name])

    server.model_aggregate(weight_accumulator=weight_accumulator)
    acc, loss = server.model_eval()

    print('\nEpoch %d, acc: %f, loss: %f\n' % (e, acc, loss))
