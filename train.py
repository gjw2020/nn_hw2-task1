import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from load_Caltech101 import get_data_loaders
from model import build_resnet18


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, early_stop_patience=5, writer=None, tag=""):
    """训练和验证函数（每个epoch输出一行：train_loss, val_loss, train_acc, val_acc）
       支持 early stopping，当 val_acc 多轮未提升时提前停止。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device on {device}')
    model = model.to(device)

    if torch.cuda.device_count() > 4:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    best_acc = 0.0
    print(f"Training {tag} model...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算指标
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        

        # 写入 TensorBoard
        if writer:
            writer.add_scalar(f'{tag}/Train Loss', train_loss, epoch)
            writer.add_scalar(f'{tag}/Val Loss', val_loss, epoch)
            writer.add_scalar(f'{tag}/Train Acc', train_acc, epoch)
            writer.add_scalar(f'{tag}/Val Acc', val_acc, epoch)

        # 单行输出（格式：epoch, train_loss, val_loss, train_acc, val_acc）
        print(f"{tag} Epoch {epoch+1:2d}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # 保存最佳模型 & 早停逻辑
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_wts = model.state_dict()
            best_model_wts = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(best_model_wts, f"resnet18_{tag}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {early_stop_patience} epochs)")
                break
        
    return best_acc


def hyperparameter_search(
    train_loader,
    val_loader,
    finetune_learning_rates,
    lr_ratio,
    scratch_learning_rates,
    num_epochs_list,
    weight_decays,
    writer=None
):
    """
    执行超参数搜索，比较微调预训练模型与从头训练模型的性能。

    参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        finetune_learning_rates: 微调学习率列表
        lr_ratio: 输出层/非输出层
        scratch_learning_rates: 全调学习率列表
        num_epochs_list: 训练轮数列表
        weight_decays: 正则参数
        writer: TensorBoard 的 SummaryWriter 实例（可选）

    返回:
        best_finetune: 微调模型的最佳性能及其参数
        best_scratch: 从头训练模型的最佳性能及其参数
    """
    best_finetune = {"acc": 0.0, "params": None, "model_path": None}
    best_scratch = {"acc": 0.0, "params": None, "model_path": None}

    # 微调预训练模型
    for lr in finetune_learning_rates:
        lr_output = lr * lr_ratio
        for wd in weight_decays:
            for num_epochs in num_epochs_list:
                tag = f"finetune_lr{lr}_fc{lr_output}_wd{wd}_ep{num_epochs}"
                model_ft = build_resnet18(pretrained=True, num_classes=101)

                # 分层正则化（conv加，bn/bias/fc不加）
                decay_params, no_decay_params = [], []
                for name, param in model_ft.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "fc" in name:
                        no_decay_params.append(param)  # fc整体不加正则
                    elif len(param.shape) == 1 or "bn" in name.lower() or name.endswith(".bias"):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                optimizer_ft = optim.SGD([
                    {"params": decay_params, "weight_decay": wd, "lr": lr},
                    {"params": no_decay_params, "weight_decay": 0.0, "lr": lr_output}
                ], momentum=0.9)

                acc_ft = train_model(
                    model_ft, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=optimizer_ft,
                    num_epochs=num_epochs,
                    writer=writer,
                    tag=tag
                )

                model_path_ft = f"resnet18_{tag}_best.pth"
                if acc_ft > best_finetune["acc"]:
                    best_finetune.update({
                        "acc": acc_ft,
                        "params": (lr, lr_output, wd, num_epochs),
                        "model_path": model_path_ft
                    })

    # 从头训练模型（分层正则化：bias/BN不加）
    for lr in scratch_learning_rates:
        for wd in weight_decays:
            for num_epochs in num_epochs_list:
                tag = f"scratch_lr{lr}_wd{wd}_ep{num_epochs}"
                model_scratch = build_resnet18(pretrained=False, num_classes=101)

                # 分组参数，bias/bn不加正则
                decay_params, no_decay_params = [], []
                for name, param in model_scratch.named_parameters():
                    if not param.requires_grad:
                        continue
                    if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                optimizer_scratch = optim.SGD([
                    {"params": decay_params, "weight_decay": wd},
                    {"params": no_decay_params, "weight_decay": 0.0}
                ], lr=lr, momentum=0.9)

                acc_scratch = train_model(
                    model_scratch, train_loader, val_loader,
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=optimizer_scratch,
                    num_epochs=num_epochs,
                    writer=writer,
                    tag=tag
                )

                model_path_scratch = f"resnet18_{tag}_best.pth"
                if acc_scratch > best_scratch["acc"]:
                    best_scratch.update({
                        "acc": acc_scratch,
                        "params": (lr, wd, num_epochs),
                        "model_path": model_path_scratch
                    })
                    
    return best_finetune, best_scratch


# 实验主流程
def main():
    # 加载数据
    train_loader, val_loader, _, _ = get_data_loaders(
        root_dir="101_ObjectCategories",
        batch_size=64,
        img_size=224
    )
    
    writer = SummaryWriter(log_dir="logs/caltech101_experiment_weightdecay")

    # 超参数搜索空间
    finetune_learning_rates = [1e-3, 1e-4]           # 微调网络非输出层的学习率
    lr_ratio = 100     # 输出层/非输出层
    scratch_learning_rates = [1e-2, 1e-3]    # 全随机网络的学习率
    num_epochs_list = [20, 30]
    weight_decays = [0.0, 1e-3]

    best_ft, best_scratch = hyperparameter_search(
        train_loader,
        val_loader,
        finetune_learning_rates,
        lr_ratio,
        scratch_learning_rates,
        num_epochs_list,
        weight_decays,
        writer = writer
    )
             

    writer.close()
    print("Best Finetune Model:", best_ft)
    print("Best From Scratch Model:", best_scratch)

if __name__ == "__main__":
    main()
