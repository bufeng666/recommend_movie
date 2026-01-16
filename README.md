# 🎬 联邦学习电影推荐系统设计文档

## 1. 项目概述

本项目构建了一个基于 **联邦学习 (Federated Learning)** 架构的电影推荐系统。系统利用 **MovieLens** 数据集，结合了深度学习推荐模型 (**NeuMF**) 与三个核心优化模块 (**CIESS**, **FedCIA**, **UniGRF**)，旨在实现高效、隐私安全且精准的个性化推荐。

系统包含完整的训练流程、推理引擎以及一个可视化的 Web 交互界面。

---

## 2. 系统架构设计

本系统目前采用**单机模拟**的方式实现了“服务器-客户端”的联邦学习架构。虽然逻辑上划分了 Server 和 Client，但在物理部署上目前仅支持单服务器运行，尚未部署多服务器分布式训练环境。

### 2.1 整体流程
1.  **初始化**: 服务器初始化全局 NeuMF 模型。
2.  **联邦训练循环 (Federated Loop)**:
    *   **客户端选择 (FedCIA)**: 服务器根据策略选择一部分活跃客户端。
    *   **本地训练**: 被选中的客户端下载全局模型，使用本地数据（模拟的数据分片）进行训练。
    *   **梯度压缩 (CIESS)**: 客户端对计算出的梯度进行稀疏化和量化，减少通信体积。
    *   **安全聚合 (FedCIA)**: 服务器接收压缩后的梯度，解压并进行加权聚合（加入差分隐私噪声），更新全局模型。
3.  **推荐服务 (UniGRF)**:
    *   训练好的模型部署到推理引擎。
    *   利用向量检索技术生成候选集，再利用模型进行精细排序。

---

## 3. 核心模块详解

### 3.1 CIESS 模块 (通信高效安全机制)
**全称**: Communication-Efficient Secure Scheme
**目标**: 解决联邦学习中客户端与服务器之间通信开销过大的问题。

*   **实现逻辑**:
    1.  **Top-k 稀疏化 (Sparsification)**: 在梯度上传前，仅保留绝对值最大的前 $k\%$ 个梯度元素，其余置为 0。这基于假设：只有大幅度的梯度更新对模型收敛最重要。
    2.  **量化 (Quantization)**: 将保留下来的浮点数梯度值量化为低精度整数（如 8-bit），进一步压缩数据体积。
    3.  **解压与重建**: 服务器端接收到索引和量化值后，重建稀疏梯度张量用于更新。
*   **集成位置**: 客户端训练流程中。
    *   在本地训练完成，准备上传梯度给服务器**之前**。
    *   它直接作用于 `NeuMF` 模型反向传播产生的梯度张量 (`grads`)。
*   **代码位置**: `src/modules/ciess.py`

### 3.2 FedCIA 模块 (联邦客户端选择与聚合)
**全称**: Federated Client Selection and Aggregation
**目标**: 优化训练效率并保护用户隐私。

*   **实现逻辑**:
    1.  **客户端选择**: 模拟了一个评分机制（基于数据质量、设备状态等），每轮训练只选择分数最高的 $N$ 个客户端参与，避免“掉队者”拖慢整体进度。
    2.  **安全聚合 (Secure Aggregation)**:
        *   **加权平均**: 根据客户端的数据量权重聚合梯度。
        *   **差分隐私 (Differential Privacy)**: 在聚合结果上添加高斯噪声 (Gaussian Noise)，确保服务器无法反推出单个用户的具体交互数据。
*   **集成位置**: 服务器端的联邦控制循环 (Federated Loop)。
    *   **选择**: 在每一轮训练**开始前**，决定分发模型给哪些客户端。
    *   **聚合**: 在接收到所有客户端回传的梯度**之后**，更新全局模型参数**之前**。
*   **代码位置**: `src/modules/fedcia.py`

### 3.3 UniGRF 模块 (统一生成式检索框架)
**全称**: Unified Generative Retrieval Framework
**目标**: 实现从海量电影库中毫秒级的精准推荐。

*   **实现逻辑 (两阶段推荐)**:
    1.  **召回阶段 (Retrieval)**:
        *   利用 **FAISS** 向量库构建索引。
        *   使用模型的 **GMF 部分** 生成 User Embedding 和 Item Embedding。
        *   通过近似最近邻搜索 (ANN)，快速从 25,000+ 部电影中筛选出 Top-100 候选集。
    2.  **排序阶段 (Ranking)**:
        *   使用完整的 **NeuMF 模型**（包含 MLP 部分）对 Top-100 候选集进行逐一打分。
        *   根据预测评分进行最终排序，输出 Top-10。
*   **集成位置**: 推理与应用阶段 (Inference Phase)。
    *   它**不参与**模型的训练过程。
    *   它作为训练好的 `NeuMF` 模型的**上层封装**，负责调用模型的 Embedding 层进行检索，调用模型的全连接层进行打分。
*   **代码位置**: `src/modules/unigrf.py`

---

## 4. 模型架构 (NeuMF)

本系统采用 **NeuMF (Neural Matrix Factorization)** 作为核心推荐算法。它融合了传统的矩阵分解和深度神经网络，能够同时捕捉线性和非线性的用户-物品交互特征。

### 4.1 模型结构
模型由两个并行分支组成，最终在输出层融合：

1.  **GMF 分支 (Generalized Matrix Factorization)**:
    *   **输入**: User ID, Item ID。
    *   **操作**: 将 User Embedding 和 Item Embedding 进行**逐元素乘积 (Element-wise Product)**。
    *   **作用**: 捕捉用户和物品之间的线性交互关系（类似传统 MF）。

2.  **MLP 分支 (Multi-Layer Perceptron)**:
    *   **输入**: User ID, Item ID。
    *   **操作**: 将 User Embedding 和 Item Embedding **拼接 (Concatenate)**。
    *   **网络**: 经过多层全连接层 (Dense Layers) + ReLU 激活 + Dropout。
        *   层级结构: `[Embedding_Dim * 2] -> [64] -> [32] -> [16]`
    *   **作用**: 捕捉用户和物品之间复杂的、非线性的交互模式。

3.  **NeuMF Layer (融合层)**:
    *   将 GMF 输出向量和 MLP 输出向量拼接。
    *   通过一个线性层 (`Linear(Embedding + 16, 1)`) 输出最终预测评分。

### 4.2 特征提取与冷启动
*   **现有用户**: 直接查找训练好的 `nn.Embedding` 矩阵获取特征向量。
*   **新用户 (冷启动)**:
    *   系统允许新用户输入几部喜欢的电影及其评分。
    *   **特征生成**: 计算用户已评分电影的 Item Embedding 的**加权平均值**。
    *   **公式**: $E_{user} = \frac{\sum (Rating_i \times E_{item\_i})}{\sum Rating_i}$
    *   系统同时计算 GMF 和 MLP 两套 Embedding，以适配 NeuMF 的双塔结构。

---

## 5. 项目文件结构

```text
d:\humou\recomend\
│  model.pth                # 训练好的模型权重
│  requirements.txt         # 项目依赖
│  PROJECT_DOCS.md          # 本文档
│
├─ml-latest/                # MovieLens 数据集
│      movies.csv
│      ratings.csv
│      ...
│
├─src/                      # 核心源代码
│  │  data_loader.py        # 数据加载与预处理
│  │  inference.py          # 命令行推理脚本
│  │  main.py               # 训练主程序 (集成 FL 模拟)
│  │  model.py              # NeuMF 模型定义
│  │
│  └─modules/               # 三大核心模块
│          ciess.py         # 梯度压缩
│          fedcia.py        # 联邦聚合
│          unigrf.py        # 检索与排序
│
└─web/                      # Web 可视化界面
    │  app.py               # Flask 后端
    └─templates/
            index.html      # 前端页面
```

## 6. 快速开始

### 6.1 环境安装
```bash
pip install -r requirements.txt
```

### 6.2 训练模型
运行主程序进行模拟联邦训练。训练结束后会自动评估并保存模型至 `model.pth`。
```bash
python src/main.py
```

### 6.3 启动 Web 界面
启动 Flask 服务器，在浏览器中进行交互式推荐。
```bash
python web/app.py
```
访问地址: `http://127.0.0.1:5000`

### 6.4 命令行预测
如果不使用 Web 界面，也可以使用命令行脚本进行测试。
```bash
python src/inference.py
```

---

## 7. 未来发展方向

### 7.1 多服务器分布式训练 (Multi-Server Training)
当前系统在单机上串行模拟联邦学习过程。未来的核心升级方向是支持真实的分布式环境：
*   **参数服务器 (Parameter Server) 架构**: 部署独立的参数服务器集群，负责全局模型的存储和聚合。
*   **多节点并行训练**: 允许不同的物理机器作为 Client 节点，并行地进行本地训练和梯度上传。
*   **通信协议升级**: 从内存交换改为基于 gRPC 或 Socket 的网络通信，实现真实的远程梯度传输。

### 7.2 异构设备支持
*   适配不同计算能力的边缘设备（如手机、IoT 设备），根据设备性能动态调整模型大小或训练负载。

### 7.3 增强的隐私保护
*   引入同态加密 (Homomorphic Encryption) 或多方安全计算 (MPC) 来替代或增强现有的差分隐私方案，提供更严格的隐私保证。
