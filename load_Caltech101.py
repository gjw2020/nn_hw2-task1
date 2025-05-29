import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

class Caltech101Dataset(Dataset):
    """Caltech-101数据集类（仅标准划分）"""
    def __init__(self, root_dir, transform=None, train=True, seed=42):
        """
        参数:
            root_dir: 数据集根目录
            transform: 数据预处理
            train: 是否为训练集（True=训练集，False=测试集）
            seed: 随机种子（控制可重复性）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.seed = seed
        
        # 加载所有图像路径和标签
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([cls for cls in os.listdir(root_dir) 
                                  if cls != "BACKGROUND_Google"])
        self.label_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # 遍历每个类别文件夹
        for label_name in self.class_names:
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
                
            # 获取当前类别所有图像
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 记录路径和标签
            for img in images:
                self.image_paths.append(os.path.join(class_dir, img))
                self.labels.append(self.label_to_idx[label_name])
        
        # 转换为numpy数组便于划分
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        
        # 执行标准划分（每类30训练图）
        self._standard_split()

    def _standard_split(self):
        """执行Caltech-101标准划分（每类30训练图）"""
        train_indices, test_indices = [], []
        np.random.seed(self.seed)
        
        # 对每个类别进行划分
        for class_idx in range(len(self.class_names)):
            class_mask = (self.labels == class_idx)
            class_indices = np.where(class_mask)[0]
            
            # 随机选择30个训练样本
            train_selected = np.random.choice(class_indices, 30, replace=False)
            train_indices.extend(train_selected)
            
            # 其余作为测试样本
            test_indices.extend(list(set(class_indices) - set(train_selected)))
        
        # 根据train参数选择子集
        if self.train:
            self.image_paths = self.image_paths[train_indices]
            self.labels = self.labels[train_indices]
        else:
            self.image_paths = self.image_paths[test_indices]
            self.labels = self.labels[test_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loaders(root_dir, batch_size=32, img_size=224, val_ratio=0.1, seed=42):
    """
    获取标准划分的DataLoader（训练/验证/测试集）
    
    参数:
        root_dir: 数据集根目录
        batch_size: 批量大小
        img_size: 图像缩放尺寸
        val_ratio: 验证集比例（从训练集划分）
        seed: 随机种子
    返回:
        train_loader, val_loader, test_loader, class_names
    """
    # 数据增强和归一化
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载标准划分的训练集和测试集
    train_dataset = Caltech101Dataset(
        root_dir=root_dir,
        transform=train_transform,
        train=True,
        seed=seed
    )
    
    test_dataset = Caltech101Dataset(
        root_dir=root_dir,
        transform=test_transform,
        train=False,
        seed=seed
    )
    
    # 从训练集划分验证集
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_ratio * num_train))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    # 验证集使用测试集的transform（关闭数据增强）
    val_subset.dataset.transform = test_transform
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names
