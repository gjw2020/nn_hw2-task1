# nn_hw2-task1

期中作业 task1，微调在 ImageNet 上预训练的卷积神经网络实现 Caltech-101 分类。

本项目旨在比较在 Caltech-101 数据集上，对 ImageNet 预训练的 ResNet-18 模型进行微调与从零开始训练 ResNet-18 模型的性能差异。

## 文件结构

- `train.py`: 包含模型训练、验证、超参数搜索以及 TensorBoard 日志记录的实现。
- `test.py`: 用于评估训练好的模型在测试集上的性能，计算总体准确率和平均类准确率 (Mean Class Accuracy, mCA)。
- `model.py`: 定义了加载预训练或随机初始化 ResNet-18 模型并修改最后全连接层的函数。
- `load_Caltech101.py`: 实现了 Caltech-101 数据集的加载和标准的训练/测试集划分，并从训练集中划分出验证集。
- `README.md`: 本文件，提供项目概览和使用说明。

## 如何使用

### 1. 下载数据集

- 您可以使用以下命令行命令下载并解压 Caltech-101 数据集：

  ```bash
  wget -O caltech-101.zip "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
  unzip caltech-101.zip
  cd caltech-101
  tar -xzvf 101_ObjectCategories.tar.gz
  ```

- 将解压后的文件夹（通常命名为 `101_ObjectCategories`）放置到本项目的根目录下（即与 `train.py`、`test.py` 等文件同级）。

### 2. 安装依赖

- 本项目依赖以下 Python 库：
  - `torch`
  - `torchvision`
  - `numpy`
  - `scikit-learn` (用于计算准确率)
  - `tensorboard` (用于可视化训练过程)
- 您可以使用 pip 安装这些依赖，请在命令行中执行以下命令：

  ```bash
  pip install torch torchvision numpy scikit-learn tensorboard
  ```

### 3. 运行训练和超参数搜索

- 在命令行中，进入项目根目录，然后执行以下命令运行训练脚本：

  ```bash
  python train.py
  ```

- 该脚本将自动加载 Caltech-101 数据集，执行超参数搜索（比较微调模型与从头训练模型），并在训练过程中生成 TensorBoard 日志文件（保存在项目根目录下的 `logs/` 文件夹中）以及最佳模型权重文件（`.pth` 文件，保存在项目根目录下）。

### 4. 查看 TensorBoard 日志

- 在训练过程中或训练完成后，您可以在项目根目录下运行以下命令启动 TensorBoard：

  ```bash
  tensorboard --logdir=logs/
  ```

- 然后在浏览器中访问 TensorBoard 提供的地址（通常是 `http://localhost:6006`），查看训练过程中的损失和准确率曲线。

### 5. 评估模型

- 修改 `test.py` 文件中的 `pth_path` 变量，指定您想要评估的模型权重文件（例如，`resnet18_finetune_lr0.001_fc0.1_wd0.0_ep30_best.pth`）。
- 在命令行中，进入项目根目录，然后执行以下命令运行评估脚本：

  ```bash
  python test.py
  ```

- 评估脚本将输出测试集上的总体准确率（Overall Accuracy）和平均类准确率（Mean Class Accuracy, mCA）。

### 6. 其他说明

- 训练过程中，如果验证集上的性能在连续多轮（默认 patience 为 5）未提升，则训练将提前停止（early stopping）。
- 超参数搜索（在 `train.py` 中）会尝试不同的学习率、学习率比例（输出层/非输出层）以及权重衰减，以比较微调模型与从头训练模型的性能差异，并在训练结束后打印出最佳模型的参数和性能。

## 数据集

项目使用 Caltech-101 数据集。`load_Caltech101.py` 脚本会自动处理数据集的加载和标准的训练/测试集划分 (每类随机选择 30 张图片作为训练集)。验证集是从训练集中按比例划分出来的。

请将数据集下载并放置在项目根目录下的 `101_ObjectCategories` 文件夹中。

## 模型

项目使用了经典的卷积神经网络模型 ResNet-18。

- `model.py` 中的 `build_resnet18` 函数负责加载模型。
- 可以通过设置 `pretrained=True` 来加载 ImageNet 预训练权重，然后只微调最后的全连接层，或者设置 `pretrained=False` 从头开始训练整个网络。

## 训练与超参数搜索

- `train.py` 实现了模型的训练和验证过程，支持 early stopping。
- `train.py` 中的 `hyperparameter_search` 函数用于进行超参数搜索，比较微调模型和从头训练模型的性能。搜索的超参数包括学习率、学习率比例 (输出层/非输出层) 和权重衰减。
- 训练过程中的损失和准确率会记录到 TensorBoard 中，方便可视化分析。日志文件将保存在项目根目录下的 `logs/` 文件夹中。

## 评估

- `test.py` 用于评估训练好的模型在测试集上的性能。
- 评估指标包括：
  - **Overall Accuracy**: 测试集上所有样本的总体准确率。
  - **Mean Class Accuracy (mCA)**: 各类别准确率的平均值，更能反映模型在类别不平衡数据集上的性能。
- 可以在 `test.py` 中指定要评估的模型权重文件 (`.pth` 文件)。
