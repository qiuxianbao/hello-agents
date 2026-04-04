"""
知识点：
PyTorch 是一个开源的【机器学习框架】，广泛用于深度学习和人工智能应用开发。

它提供了两个核心功能：
张量计算（类似 NumPy，但支持 GPU 加速）
自动微分（自动计算梯度，用于神经网络训练）

运行设备：GPU
"""
from pathlib import Path

import torch
import torch.nn as nn  # nn（Neural Network） 神经网络

current_dir = Path(__file__).parent


class NeuralNetwork(nn.Module):
    """
    nn.Module   所有 PyTorch 模型的基类
    """

    def __init__(self):
        super().__init__()
        # 展平层
        self.flatten = nn.Flatten()  # 将多维数据展平为一维

        #  3 个全连接层（784→512→128→10）
        # 使用 ReLU 激活函数增加非线性
        self.layers = nn.Sequential(
            # nn.Linear 全连接层（线性变换）
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def test_torch_tensor():
    """
    张量（Tensor）就是“多维数组”。它是数据的容器，可以存储数字并进行数学运算

    你可以根据它的“维数”（也叫阶或秩）来直观理解
    0 维张量：标量（Scalar），就是一个单一的数字，如 5。
    1 维张量：向量（Vector），一排数字，如 [1, 2, 3]。
    2 维张量：矩阵（Matrix），有行有列的数字方阵
    :return:
    """
    one = torch.tensor(1)
    print(one)  # tensor(1)

    tensor = torch.tensor([1, 2, 3])
    print(tensor)  # tensor([1, 2, 3])

    zeros = torch.zeros(3, 3)
    """
    tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
    """
    print(zeros)

    ones = torch.ones(2, 2)
    """
    tensor([[1., 1.],
        [1., 1.]])
    """
    print(ones)

    # math:[0, 2)
    random_tensor = torch.rand(3, 3)
    print(random_tensor)

    # 张量运算
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    sum_result = a + b
    # [5, 7, 9]
    print(sum_result)

    mul_result = a * b
    # [4, 10, 28]
    print(mul_result)

    # 生成一个服从标准正态分布（均值为 0，方差为 1）的随机张量
    # (128, 20) - 张量的形状
    #   128：第一个维度（行）
    #       通常代表 batch size（批量大小）
    #       一次处理 128 个样本
    #   20：第二个维度（列）
    #       通常代表 特征数量（input features）
    #       每个样本有 20 个特征
    # 随机模拟 128 个样本，每个样本 20 个特征
    batch_input = torch.randn(128, 20)
    # 均值
    # print(batch_input.mean())  # 接近 0
    # 样本标准差
    # print(batch_input.std())  # 接近 1

    # 用于测试神经网络
    """
    我们可以把 nn.Linear(20, 10) 拆解为它内部存储的两个“零件”：
    权重 (Weight)：一个形状为 [10, 20] 的矩阵。
    偏置 (Bias)：一个长度为 10 的向量

    第一步：相乘（特征组合）
    第二步：相加（调整偏移）
    """
    model = nn.Linear(20, 10)  # 输入 20 维，输出 10 维
    output = model(batch_input)  # 前向传播
    print(output.shape)  # torch.Size([128, 10])


def test_torch_gpu():
    """
    PyTorch 支持 GPU 加速，你可以将张量移到 GPU 上进行运算。
    """
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 将张量移到 GPU
    tensor = torch.tensor([1, 2, 3])
    tensor_gpu = tensor.to(device)
    print(tensor_gpu)


def test_torch_nn():
    """
    完整的 PyTorch 神经网络使用示例
    
    这个 demo 展示了从创建模型到训练、评估的完整流程
    """
    print("=" * 60)
    print("PyTorch 神经网络完整使用示例")
    print("=" * 60)

    # 步骤 1: 构建神经网络
    print("\n【步骤 1】构建神经网络")
    model = NeuralNetwork()
    print(f"✓ 模型已创建：{model.__class__.__name__}")
    """
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layers): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=128, bias=True)
        (3): ReLU()
        (4): Linear(in_features=128, out_features=10, bias=True)
      )
    )
    """
    print(f"  模型结构:\n{model}")

    # 步骤 2: 准备输入数据
    print("\n【步骤 2】准备输入数据")
    # 创建 2 张 28x28 像素的灰度图像（随机噪声）
    # 图像数据 (batch_size=2, 通道=1, 高/行=28, 宽/列=28)
    sample_image = torch.randn(2, 1, 28, 28)
    print(f"✓ 输入数据形状：{sample_image.shape}")
    print(f"  含义：batch_size=2, channels=1, height=28, width=28")

    # 步骤 3: 前向传播（预测）
    print("\n【步骤 3】前向传播（使用模型进行预测）")

    with torch.no_grad():  # 不需要计算梯度，节省内存
        output = model(sample_image)
    # torch.Size([2, 10])
    print(f"✓ 输出结果形状：{output.shape}")
    print(f"  含义：batch_size=2, num_classes=10")
    """
    tensor([[ 0.0583,  0.1493,  0.1209,  0.1583, -0.0533, -0.1025, -0.0420, -0.1039,
          0.0407, -0.0889],
        [ 0.0880, -0.0183,  0.0615,  0.0539,  0.0101, -0.1029, -0.1771, -0.1874,
         -0.0071, -0.1153]])
    """
    print(f"  原始输出 (logits):\n{output}")

    # 步骤 4: 应用 Softmax 获取概率
    print("\n【步骤 4】转换为概率分布")
    # tensor([[0.1040, 0.1140, 0.1108, 0.1150, 0.0931, 0.0886, 0.0941, 0.0885, 0.1022,
    #          0.0898],
    #         [0.1131, 0.1017, 0.1101, 0.1093, 0.1046, 0.0934, 0.0868, 0.0859, 0.1028,
    #          0.0923]])
    probabilities = torch.softmax(output, dim=1)  # dim=1 表示在类别维度上计算
    print(f"✓ 概率分布:\n{probabilities}")

    #  每行的和应该为 1: tensor([1.0000, 1.0000])
    print(f"  每行的和应该为 1: {probabilities.sum(dim=1)}")

    # 步骤 5: 获取预测类别
    print("\n【步骤 5】获取预测结果")
    # 使用 argmax 找出概率最大的类别作为预测结果
    predicted_classes = torch.argmax(probabilities, dim=1)
    # tensor([3, 0])
    print(f"✓ 预测的类别：{predicted_classes}")

    # 样本 1 预测为类别 3
    print(f"  样本 1 预测为类别 {predicted_classes[0].item()}")
    # 样本 2 预测为类别 0
    print(f"  样本 2 预测为类别 {predicted_classes[1].item()}")

    # 步骤 6: 定义损失函数和优化器
    print("\n【步骤 6】设置训练组件")

    """
    # 损失函数
    损失函数（Loss Function） 是用来衡量模型预测值与真实值（标准答案）之间“不一致”程度的数学公式
    1.回归问题：MSE（Mean Squared Error） 均方误差
    2.平均问题：
    """
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（用于分类问题）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam（学习率 0.001），用于更新模型参数
    print(f"✓ 损失函数：CrossEntropyLoss")
    print(f"✓ 优化器：Adam (learning_rate=0.001)")

    # 步骤 7: 模拟训练步骤
    """
    核心思想：
    定义网络 → 前向传播（预测）→ 计算损失 
                              ↓
    更新参数 ← 使用梯度 ← 反向传播（计算梯度）
    """
    print("\n【步骤 7】模拟训练过程")
    # 准备一些假标签（假设真实类别是 [3, 7]）
    labels = torch.tensor([3, 7])
    # 真实标签：tensor([3, 7])
    print(f"  真实标签：{labels}")

    # 7.1 前向传播,计算损失
    outputs = model(sample_image)
    loss = criterion(outputs, labels)  # 计算损失
    # 初始损失：2.3089
    print(f"  初始损失：{loss.item():.4f}")

    # 清空旧梯度
    optimizer.zero_grad()

    # 7.2 反向传播，更新参数
    loss.backward()
    optimizer.step()  # 更新模型参数
    print(f"✓ 完成一次训练迭代")

    # 再次计算损失（应该减小）
    new_outputs = model(sample_image)
    new_loss = criterion(new_outputs, labels)
    # 训练后损失：1.5026
    print(f"  训练后损失：{new_loss.item():.4f}")

    # 步骤 8: 模型评估模式
    print("\n【步骤 8】切换到评估模式")
    model.eval()  # 设置为评估模式（关闭 Dropout 等层）
    with torch.no_grad():
        eval_output = model(sample_image)
        eval_probs = torch.softmax(eval_output, dim=1)
        eval_pred = torch.argmax(eval_probs, dim=1)
    # tensor([3, 7])
    print(f"✓ 评估模式下的预测：{eval_pred}")
    # 置信度：tensor([0.2570, 0.1928])
    # 置信度 = 模型的"信心指数"，数值范围：0 到 1（或 0% 到 100%）
    print(f"  置信度：{eval_probs.max(dim=1).values}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


def test_torch_grad():
    #  PyTorch 的自动求导（autograd）功能，自动微分功能
    """
    # 梯度：
    梯度是一个向量，表示多元函数在某一点处变化最快的方向和变化率。
    它指向函数值增长最快的方向，其模（长度）表示增长的速率

    # 梯度消失：
    在深层神经网络训练过程中，当使用反向传播算法计算梯度时，梯度值会随着层数加深而变得越来越小，最终趋近于零的现象

    产生原因：
    链式法则的连乘效应。如果每个偏导数都小于 1，连乘后会更小
    :return:
    """
    # requires_grad=True 表示需要追踪这个张量的梯度，用于反向传播
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2  # 定义运算。对 x 进行平方运算，得到 y = [1.0, 4.0, 9.0]
    loss = y.sum()  # 定义损失函数，计算y的和，得到 loss = 14.0

    """
    执行反向传播，自动计算 loss 对 x 的梯度
    计算 ∂loss/∂x = 2x = [2., 4., 6.]
    
    过程：
    x = [1.0, 2.0, 3.0]
    y = x² = [1.0, 4.0, 9.0]
    loss = sum(y) = 1 + 4 + 9 = 14.0

    步骤 1：计算 ∂loss/∂y
    loss = y₁ + y₂ + y₃
    ∂loss/∂y₁ = 1
    ∂loss/∂y₂ = 1
    ∂loss/∂y₃ = 1
    
    所以：∂loss/∂y = [1, 1, 1]
    
    步骤2：计算 计算 ∂y/∂x
    y = x²
    根据求导法则：dy/dx = 2x
    
    对于每个元素：
    ∂y₁/∂x₁ = 2 × 1.0 = 2.0
    ∂y₂/∂x₂ = 2 × 2.0 = 4.0
    ∂y₃/∂x₃ = 2 × 3.0 = 6.0
    
    所以：∂y/∂x = [2.0, 4.0, 6.0]
    
    步骤 3：应用链式法则（Chain Rule）
    ∂loss/∂x = ∂loss/∂y × ∂y/∂x
    
    ∂loss/∂x₁ = ∂loss/∂y₁ × ∂y₁/∂x₁ = 1 × 2.0 = 2.0
    ∂loss/∂x₂ = ∂loss/∂y₂ × ∂y₂/∂x₂ = 1 × 4.0 = 4.0
    ∂loss/∂x₃ = ∂loss/∂y₃ × ∂y₃/∂x₃ = 1 × 6.0 = 6.0
    
    最终结果：∂loss/∂x = [2.0, 4.0, 6.0]
    这就是 x.grad 的值！✅
    """
    loss.backward()

    # 获取梯度
    print(x.grad)  # [2., 4., 6.]


def test_torch_dataloader():
    """
        原始数据                    DataLoader 处理                输出批次
    ┌──────────────┐          ┌─────────────┐            ┌─────────────┐
    │ features     │          │             │            │ Batch 1     │
    │ [100, 784]   │          │   打乱顺序   │            │ [32, 784]   │
    │              │          │             │            ├─────────────┤
    │ labels       │    →     │   分批处理   │    →       │ Batch 2     │
    │ [100]        │          │             │            │ [32, 784]   │
    └──────────────┘          │             │            ├─────────────┤
                              │             │            │ Batch 3     │
                              │             │            │ [32, 784]   │
                              └─────────────┘            ├─────────────┤
                                                         │ Batch 4     │
                                                         │ [4, 784]    │
                                                         └─────────────┘

    :return:
    """
    # 数据加载
    from torch.utils.data import DataLoader, TensorDataset

    # 创建模拟数据
    features = torch.randn(100, 784)  # 100 个样本，每个样本 784 个特征
    # 结果形状 [100]
    labels = torch.randint(0, 20, (100,))  # 100 个标签，范围 0-19

    # 将特征和标签打包成一个数据集对象
    dataset = TensorDataset(features, labels)
    # 每批加载 32 个样本
    # 每个 epoch 打乱数据顺序，防止模型记住数据顺序，提高泛化能力
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据加载器
    """
    # 第一次迭代
    batch_features.shape = [32, 784]  # 32 个样本，每个 784 特征
    batch_labels.shape   = [32]       # 32 个标签
    
    # 第二次迭代
    batch_features.shape = [32, 784]
    batch_labels.shape   = [32]
    
    # 第三次迭代
    batch_features.shape = [32, 784]
    batch_labels.shape   = [32]
    
    # 第四次迭代（最后一批）
    batch_features.shape = [4, 784]   # 只剩 4 个样本
    batch_labels.shape   = [4]
    """
    for batch_features, batch_labels in dataloader:
        # Batch shape: torch.Size([32, 784]), Labels shape: torch.Size([32])
        print(f"Batch shape: {batch_features.shape}, Labels shape: {batch_labels.shape}")
        break  # 只打印第一个 batch


def test_torch_save():
    # 模型保存和加载
    import os

    # 创建并保存模型
    model = NeuralNetwork()
    model_path = 'model.pth'  # .pth：PyTorch 模型的标准文件扩展，其他常见格式：.pt, .bin, .onnx

    # PyTorch 的保存函数，使用 pickle 序列化
    torch.save(model.state_dict(), model_path)  # 模型的参数字典（不包含网络结构）
    print(f"模型已保存到 {model_path}")

    # 加载模型
    model_loaded = NeuralNetwork()
    # 从文件中读取参数， weights_only 只加载张量数据，不执行 Python代码，防止恶意 pickle 攻击
    saved_params = torch.load(model_path, weights_only=True)
    # 将参数加载到模型中
    model_loaded.load_state_dict(saved_params)

    # 设置为评估模式
    """
    train()                         eval()
    模型训练阶段                      模型预测/推理阶段
    创建模型后，默认                   需要手动设置
    开启（需要反向传播）                通常关闭（节省内存）
    """
    model_loaded.eval()
    print("模型已成功加载")

    # 清理文件
    if os.path.exists(model_path):
        os.remove(model_path)


def test_torch_layer():
    """
    卷积层(卷：将其中一个函数进行翻转（像卷轴一样折叠）；积：这是指乘积运算与累加)
    理解：像是在用放大镜仔细扫描每一个细节（边缘、纹理）

    作用：
    从输入图像中提取局部特征（边缘、纹理、形状等）
    通过滑动窗口（卷积核）在图像上扫描，检测特定模式

    参数解读：
    in_channels=3: 输入 3 通道（RGB 彩色图像）
    out_channels=64: 输出 64 个特征图（学习 64 种不同的特征）
    kernel_size=3: 卷积核大小 3×3（感受野）

    感受野（Receptive Field, RF）是指神经网络中每一层输出的特征图（Feature Map）上的一个像素点，在原始输入图像上映射的区域大小

    使用场景：
    ✅ 图像分类（识别猫/狗/车）
    ✅ 目标检测（找到图中的人脸位置）
    ✅ 图像分割（医学影像分析）
    ✅ 风格迁移（把照片变成梵高风格

    :return:
    """
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
    # 输入：一张 224x224 的 RGB 图片
    input_image = torch.randn(1, 3, 224, 224)  # (batch, channels, height, width)
    output = conv(input_image)
    # torch.Size([1, 64, 222, 222])
    print(output.shape)

    """
    池化层(其核心含义是“汇集”或“合用”.)
    理解：让你眯起眼睛，把一小片区域看成一个色块。你损失了精确位置，但换取了对整体轮廓更宏观、更稳定的把握
    
    作用：
        降维：减小特征图尺寸，降低计算量
        保留显著特征：提取局部区域的最大值
        防止过拟合：一定程度上做信息压缩
        
    参数解读：
        kernel_size=2: 2×2 的窗口进行下采样（尺寸缩小为原来 1/2）

    使用场景：
    ✅ CNN 网络中：通常紧跟在卷积层之后
    ✅ 需要减少参数量时：加快训练速度
    ✅ 提取主要特征：忽略细微噪声
    """
    pool = nn.MaxPool2d(kernel_size=2)
    # 输入：卷积层输出的特征图 (64, 222, 222)
    feature_map = torch.randn(1, 64, 222, 222)
    # 空间尺寸减半，通道数不变
    pooled = pool(feature_map)  # 输出：(1, 64, 111, 111)
    print(pooled.shape)

    """
    Dropout
    """
    dropout = nn.Dropout(0.5)

    """
    BatchNorm
    """
    bn = nn.BatchNorm2d(64)

    """
    RNN/LSTM
    """
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)


def test_nn_activation_func():
    """
    激活函数
    是给神经网络装上“开关”和“弯道”，让它不再是一根筋的直线，而是能处理复杂、拐弯抹角的问题
    :return:
    """

    # Rectified Linear Unit，修正线性单元,（增加非线性）
    # 只要输入是负数，输出直接变成 0，这个神经元就完全“死”掉了，不再起作用
    # f(x) = max(0, x)
    relu = nn.ReLU()
    # 单个值
    print(relu(torch.tensor(-0.5)))  # tensor(0.)
    print(relu(torch.tensor(0)))  # tensor(0.)
    print(relu(torch.tensor(0.5)))  # tensor(0.5000)

    # 矩阵
    print(relu(torch.tensor([-0.5, 0, 0.5])))  # tensor([0.0000, 0.0000, 0.5000])

    # Gaussian Error Linear Unit，高斯误差线性单元
    # 输入越小，越应该被丢弃（变成 0）；输入越大，越应该保留
    # 但在 0 附近，它不会像 ReLU 那样咔嚓一刀切断，而是画出一条平滑的曲线
    gelu = nn.GELU()
    print(gelu(torch.tensor(-0.5)))  # tensor(-0.1543)
    print(gelu(torch.tensor(0.0)))  # tensor(0.0)   接收的是浮点数
    print(gelu(torch.tensor(0.5)))  # tensor(0.3457)

    # 这两个函数就像是神经网络最后的“翻译官”，负责把一堆乱七八糟的计算结果（分数值），转换成人类能看懂的“概率”（0到1之间的数）
    # 1. Sigmoid：二选一（是非题），它把任何数字都压缩到 0 到 1 之间
    # f(x) =  1 / (1 + torch.exp(-x))
    sigmoid = nn.Sigmoid()
    print(sigmoid(torch.tensor(5)))  # tensor(0.9933)

    # Hyperbolic Tangent, 双曲正切函数
    # 如果说 Sigmoid 是把数值压缩到 0 到 1 之间，那么 Tanh 就是它的“增强版”，把数值压缩到 -1 到 1 之间
    # tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    tanh = nn.Tanh()
    print(tanh(torch.tensor(5)))  # tensor(0.9999)

    # 指数化归一
    # 把一堆随意的分数值（Logits），转换成一组总和为 100% 的概率分布
    # Softmax(z_i) = e^z_i / Σ_j e^z_j
    # 分子（指数化）：给每个分数取指数，确保结果全是正数，且差距被放大（高的更高，低的更低）
    # 分母（归一化）：所有分子的总和。用分子除以分母，就能保证所有结果加起来刚好等于 1
    softmax = nn.Softmax(dim=1)
    input_tensor = torch.tensor([
        [1.0, 2.0, 3.0],  # 样本 1: 3个类别的得分
        [3.0, 1.0, 2.0],  # 样本 2: 3个类别的得分
    ])

    output = softmax(input_tensor)
    # tensor([[0.0900, 0.2447, 0.6652], # 样本1的概率
    #         [0.6652, 0.0900, 0.2447]])  # 样本 2的概率
    print(output)


def test_torch_activation_func():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 定义各种激活函数
    def relu(x):
        return torch.maximum(x, torch.tensor(0.0))

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def tanh(x):
        return torch.tanh(x)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2行3列 fig是画布，axes是各个小格子
    fig.suptitle('常见激活函数可视化对比', fontsize=16, fontweight='bold')

    # x 轴范围
    # 生成从 -6 到 6 的 1000 个等间距数值
    x = torch.linspace(-6, 6, 1000)

    # 1. ReLU
    # 绘制 y 轴范围
    # 将 PyTorch 张量转为 NumPy 数组（Matplotlib 需要）
    # 参数：
    #   x.numpy() x轴数据
    #   relu(x).numpy() y轴数据
    # 'g-'：绿色实线（g=green，-=实线）
    axes[0, 0].plot(x.numpy(), relu(x).numpy(), 'g-', linewidth=2, label='ReLU')

    # X轴Y轴线的风格
    # Axes Vertical Line（垂直线）
    # x=0：在 x=0 处绘制（y 轴位置）
    # alpha=0.3：透明度 30%（半透明）
    axes[0, 0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    # axhline：Axes Horizontal Line（水平线）
    axes[0, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # 填充区域
    # 第一条曲线：y=0（x 轴）
    # 第二条曲线：relu(x)（ReLU 函数曲线）
    axes[0, 0].fill_between(x.numpy(), 0, relu(x).numpy(), alpha=0.3)

    axes[0, 0].set_title('ReLU 函数\nf(x) = max(0, x)', fontsize=12)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')

    # 图例
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)  # 显示网格
    axes[0, 0].set_ylim(-0.5, 6.5)  # 设置 y 轴范围

    # 2. GELU
    axes[0, 1].plot(x.numpy(), gelu(x).numpy(), 'c-', linewidth=2, label='GELU')

    axes[0, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 1].fill_between(x.numpy(), 0, gelu(x).numpy(), alpha=0.3)

    axes[0, 1].set_title('GELU 函数\nf(x) = 0.5x(1+tanh(√(2/π)(x+0.044715x³)))', fontsize=12)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('f(x)')

    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-1, 6)

    # 3. Sigmoid
    axes[0, 2].plot(x.numpy(), sigmoid(x).numpy(), 'b-', linewidth=2, label='Sigmoid')

    axes[0, 2].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='σ(0)=0.5')
    axes[0, 2].axhline(y=1, color='gray', linestyle='--', alpha=0.3)

    axes[0, 2].fill_between(x.numpy(), 0, sigmoid(x).numpy(), alpha=0.3)

    axes[0, 2].set_title('Sigmoid 函数\nσ(x) = 1/(1+e^(-x))', fontsize=12)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('σ(x)')

    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(-0.0, 1.1)

    # 4. Tanh
    axes[1, 0].plot(x.numpy(), tanh(x).numpy(), 'r-', linewidth=2, label='Tanh')

    axes[1, 0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='tanh(0)=0')
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=-1, color='gray', linestyle='--', alpha=0.3)

    axes[1, 0].fill_between(x.numpy(), 0, tanh(x).numpy(), alpha=0.3)

    axes[1, 0].set_title('Tanh 函数\nf(x) = (e^x - e^(-x))/(e^x + e^(-x))', fontsize=12)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('f(x)')

    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-1.2, 1.2)

    # 5. Softmax (一维示例)
    x_softmax = torch.linspace(-3, 3, 100)
    """
    假设 x_softmax = 2 时：
    类别 1（猫）：得分 =  2.0
    类别 2（狗）：得分 = -2.0
    类别 3（鸟）：得分 =  1.0
    
    经过 Softmax 后：
    [0.66, 0.02, 0.32]
         ↑    ↑    ↑
       66%  2%   32%  ← 概率分布
       
    也就是说 对于同一个x，三条线的y值累加 等于1   
    """
    # torch.stack()：将多个张量堆叠成一个新张量
    x2 = torch.stack([x_softmax, -x_softmax, x_softmax * 0.5], dim=1)  # 3 个输入
    softmax_out = torch.softmax(x2, dim=1)

    axes[1, 1].plot(x_softmax.numpy(), softmax_out[:, 0].numpy(), 'r-', linewidth=2, label='类别 1')
    axes[1, 1].plot(x_softmax.numpy(), softmax_out[:, 1].numpy(), 'g-', linewidth=2, label='类别 2')
    axes[1, 1].plot(x_softmax.numpy(), softmax_out[:, 2].numpy(), 'b-', linewidth=2, label='类别 3')

    axes[1, 1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].axhline(y=1 / 3, color='gray', linestyle='--', alpha=0.3, label='均匀分布')

    axes[1, 1].set_title('Softmax 函数\n(多类别概率分布)', fontsize=12)
    axes[1, 1].set_xlabel('输入值 x')
    axes[1, 1].set_ylabel('概率')

    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 2)

    plt.tight_layout()
    # 保存图片
    plt.savefig(current_dir.joinpath('test_torch_activation.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 打印关键数值对比
    print("=" * 60)
    print("激活函数关键数值对比表")
    print("=" * 60)

    # {} 占位符
    # : 格式化控制符的开始标记
    # < 左对齐
    # 8 最小宽度为8个字符
    print(f"{'x 值':<8} {'ReLU':<8} {'GELU':<10} {'Sigmoid':<10} {'Tanh':<10} {'SoftMax':<12} ")
    print("-" * 60)

    test_values = [-2.0, -1.0, -0.5, 0.0, 2.5, 1.0, 2.0]
    for val in test_values:
        x_test = torch.tensor(val)

        rel = relu(x_test).item()
        gel = gelu(x_test).item()
        sig = sigmoid(x_test).item()
        tan = tanh(x_test).item()

        print(f"{val:<8.1f} {rel:<8.1f} {gel:<10.4f}  {sig:<10.4f} {tan:<10.4f}")

    print("=" * 60)


def _main():
    # test_torch_activation_func()
    # test_nn_activation_func()

    test_torch_tensor()
    # test_torch_gpu()

    # test_torch_nn()
    # test_torch_grad()

    # test_torch_dataloader()
    # test_torch_save()

    test_torch_layer()


if __name__ == "__main__":
    _main()
