import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.functional as F
from utils.model_manager import ModelManager

import Net.ResNet20 as ResNet20

# 选择是否是随机初始化训练还是从外部导入参数
isInit = False
path = "./model/2024-11-20_08_03_26.pt"


# 检查可用 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 调试：打印所有可见 GPU 信息
if torch.cuda.is_available():
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, capability {props.major}.{props.minor}")
# 数据预处理
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# 下载并加载CIFAR-10数据集
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# 选择重新初始化还是从path读取
def load_model(isInit, path):
    temp = ResNet20.make_ResNet20()
    if isInit:
        return temp
    else:
        temp.load_state_dict(torch.load(path))
        return temp


# 创建ResNet20模型
model = load_model(isInit, path)
model = model.to(device)

# 创建管理工具
model_manager = ModelManager()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# 训练和评估函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    print(f"Train Epoch: {epoch} Average Loss: {running_loss / len(train_loader):.6f}")


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
    )
    return accuracy


# 训练模型
num_epochs = 100

max_acc = 0
max_acc_name = ""

for epoch in range(1, num_epochs + 1):
    train(model, device, trainloader, optimizer, criterion, epoch)
    accuracy = test(model, device, testloader, criterion)
    if accuracy > max_acc:
        max_acc = accuracy
        max_acc_name = model_manager.save_to_disk(model)
    scheduler.step()
    print(f"max_acc={max_acc},name:{max_acc_name}.pt")


print(f"max_acc={max_acc},name:{max_acc_name}.pt")
