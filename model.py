import torch.nn as nn
from torchvision import models


def build_resnet18(pretrained=True, num_classes=101):
    """加载ResNet-18并替换输出层"""
    model = models.resnet18(pretrained=pretrained)
    # 替换最后一层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model