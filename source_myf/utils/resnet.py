import torchvision.models as models

# 加载预训练的 ResNet50
resnet = models.resnet50(pretrained=True)

# 移除分类头，仅用作特征提取器
from torch.nn import Sequential
feature_extractor = Sequential(*list(resnet.children())[:-1])
