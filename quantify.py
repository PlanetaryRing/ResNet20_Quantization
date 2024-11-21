import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.model_manager import ModelManager

# 引入自定义的块
import Net.ResNet20 as ResNet20
import Net.QuantizedResNet20 as QResNet20
from Net.QuantizedConv2d import QuantizedConv2d as QConv2d

# 常量定义
IS_SAVE_QUANTIZED_MODEL=False
QUANTIZED_MODEL_NAME="quantized_model_2024-11-20_08_09_47"


# 定义需要用到的工具函数
def copy_conv_weights(src_conv, dst_conv):
    # 将非量化模型的卷积层权重复制到量化模型的卷积层
    # 此处涉及到参数类型的转换,由量化后的块提供转换方法
    dst_conv.copy_from_full_precision(src_conv)


def copy_bn_weights(src_bn, dst_bn):
    # 复制BatchNorm2d层的权重
    # 此处无量化操作,所以直接复制参数
    dst_bn.weight.data = src_bn.weight.data.clone()
    dst_bn.bias.data = src_bn.bias.data.clone()
    dst_bn.running_mean.data = src_bn.running_mean.data.clone()
    dst_bn.running_var.data = src_bn.running_var.data.clone()


def copy_weights(src_module, dst_module):
    for (src_name, src_child), (dst_name, dst_child) in zip(
        src_module.named_children(), dst_module.named_children()
    ):
        # 对卷积层进行量化
        if isinstance(src_child, nn.Conv2d) and isinstance(dst_child, QConv2d):
            copy_conv_weights(src_child, dst_child)
        elif isinstance(src_child, nn.BatchNorm2d) and isinstance(
            dst_child, nn.BatchNorm2d
        ):
            copy_bn_weights(src_child, dst_child)
        elif isinstance(src_child, nn.Linear) and isinstance(dst_child, nn.Linear):
            # 全连接层无量化操作
            dst_child.weight.data = src_child.weight.data.clone()
            if src_child.bias is not None:
                dst_child.bias.data = src_child.bias.data.clone()
        else:
            # 递归遍历子模块
            copy_weights(src_child, dst_child)


model_q = QResNet20.make_QuantizedResNet20()
model_std = ResNet20.make_ResNet20()
model_std.load_state_dict(torch.load("./model/2024-11-20_08_09_47.pt"))
# 调用遍历函数进行参数迁移
copy_weights(model_std, model_q)

############### 测试量化后的模型 ###############

# 数据预处理与数据加载
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 推理阶段使用cpu即可
device = torch.device("cpu")
print(f"running on {device}")

# 交叉熵作为损失函数
criterion = nn.CrossEntropyLoss()

# 测试函数
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


print("Test the quantized model:")
print(test(model_q, device, testloader, criterion))
print("Test the std model:")
print(test(model_std, device, testloader, criterion))

if IS_SAVE_QUANTIZED_MODEL:
    ModelManager().save_to_disk(model_q,QUANTIZED_MODEL_NAME)