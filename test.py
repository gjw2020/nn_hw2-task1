import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from load_Caltech101 import get_data_loaders
from model import build_resnet18
import os


# 每类准确率（Mean Class Accuracy）
def mean_class_accuracy(y_true, y_pred, class_names):
    per_class_acc = []
    for i in range(len(class_names)):
        indices = [j for j, y in enumerate(y_true) if y == i]
        if indices:
            correct = sum(1 for j in indices if y_pred[j] == i)
            per_class_acc.append(correct / len(indices))
    return sum(per_class_acc) / len(per_class_acc)


# 评估测试集
def evaluate_model(model, test_loader, class_names, pth_path=None):
    if pth_path and not os.path.isfile(pth_path):
        print(f"模型权重文件 {pth_path} 不存在，请检查文件路径或重新训练模型。")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if pth_path:
        model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    # Overall Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%")

    # Mean Class Accuracy
    mca = mean_class_accuracy(y_true, y_pred, class_names)
    print(f"Mean Class Accuracy (mCA): {mca*100:.2f}%")


if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_names = get_data_loaders("101_ObjectCategories")
    model = build_resnet18()
    pth_path = 'resnet18_finetune_lr0.001_fc0.1_wd0.0_ep30_best.pth'
    evaluate_model(model, test_loader, class_names, pth_path=pth_path)