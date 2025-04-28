# Entity-Enhanced HGCN

这个项目是对原始HGCN的扩展，通过集成实体信息和知识图谱嵌入来增强文本表示。我们已经完成了以下功能的实现：

## 已完成的功能

### 1. 数据预处理
- [x] 从exAAPD文献数据集中提取实体（NER）
- [x] 使用Wikipedia2Vec获取实体的300维向量表示
- [x] 将实体向量整合到训练数据中

### 2. 模型增强
- [x] 修改了HGCN数据结构以支持实体向量
- [x] 实现了EntityEnhancedHGCN模型
- [x] 添加了注意力机制来融合文本和实体信息
- [x] 集成了BERT作为文本编码器

### 3. 训练流程调整
- [x] 支持实体向量的批处理
- [x] 实现了新的数据加载器
- [x] 添加了模型评估功能
- [x] 支持GPU加速训练

## 数据集和预训练模型

### 1. 下载数据集
```bash
# 创建数据目录
mkdir -p data/exaapd

# 下载exAAPD数据集
wget https://github.com/YOUR_USERNAME/exaapd-dataset/releases/download/v1.0/exaapd.zip
unzip exaapd.zip -d data/exaapd/

# 或者使用curl
curl -L https://github.com/YOUR_USERNAME/exaapd-dataset/releases/download/v1.0/exaapd.zip -o exaapd.zip
unzip exaapd.zip -d data/exaapd/
```

### 2. 下载预训练模型

#### BERT模型
```bash
# 创建模型目录
mkdir -p models/bert

# 下载BERT预训练模型
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('bert-base-uncased', cache_dir='models/bert'); AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='models/bert')"
```

#### Wikipedia2Vec模型
```bash
# 创建Wiki2Vec目录
mkdir -p models/wiki2vec

# 下载Wiki2Vec预训练模型
python src/download_wiki2vec.py --output_dir models/wiki2vec
```

### 3. 验证下载
```bash
# 检查数据集文件
ls data/exaapd/
# 应该看到：train.json, dev.json, test.json等文件

# 检查模型文件
ls models/bert/
ls models/wiki2vec/
# 应该看到相应的模型文件
```

## 项目结构

```
HGCN/
├── datasets/
│   └── bert_processors/
│       ├── abstract_processor.py
│       ├── entity_enhanced_processor.py    # 新增：处理实体数据
│       └── ner_processor.py
├── models/
│   └── entity_enhanced_hgcn.py            # 新增：支持实体的HGCN模型
├── src/
│   ├── download_wiki2vec.py               # 下载Wiki2Vec模型
│   └── process_entity_vectors.py          # 处理实体和向量
├── prepare_training_data.py               # 准备训练数据
└── train_entity_enhanced.py               # 训练脚本
```

## 环境要求

```bash
Python 3.7+
PyTorch 1.8+
transformers
numpy
tqdm
Wikipedia2Vec
scikit-learn
```

## 安装步骤

1. 克隆仓库：
```bash
git clone <repository_url>
cd HGCN
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行步骤

### 1. 准备数据和模型
首先确保您已经下载了所需的数据集和预训练模型（见上述"数据集和预训练模型"部分）。

### 2. 处理实体和向量
```bash
python src/process_entity_vectors.py \
    --data_dir data/exaapd \
    --max_seq_length 128 \
    --wiki2vec_model_path models/wiki2vec/enwiki_20180420_300d.pkl
```

### 3. 准备训练数据
```bash
python prepare_training_data.py
```

### 4. 训练模型
```bash
python train_entity_enhanced.py \
    --data_dir data/exaapd_processed \
    --model bert-base-uncased \
    --max_seq_length 128 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --hidden_dim 768 \
    --dropout 0.1 \
    --num_labels 3 \
    --save_path checkpoints/entity_enhanced
```

## 主要修改内容

### 1. EntityEnhancedProcessor
- 支持实体向量的加载和处理
- 实现了数据转换和批处理功能
- 处理变长实体序列

### 2. EntityEnhancedHGCN
- 集成BERT作为文本编码器
- 添加实体向量处理模块
- 实现注意力机制融合不同类型的信息
- 支持批处理和GPU加速

### 3. 训练流程
- 支持实体向量的批处理
- 添加模型评估功能
- 实现最佳模型保存
- 支持GPU训练

## 参数说明

- `--data_dir`: 数据目录
- `--model`: 使用的预训练模型
- `--max_seq_length`: 最大序列长度
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--hidden_dim`: 隐层维度
- `--dropout`: Dropout率
- `--num_labels`: 标签数量
- `--save_path`: 模型保存路径

## 性能优化建议

1. 模型参数调整：
   - 根据GPU内存调整batch_size
   - 调整learning_rate以获得更好的收敛
   - 通过hidden_dim控制模型容量
   - 使用dropout防止过拟合

2. 数据处理优化：
   - 根据具体任务调整实体提取阈值
   - 考虑使用领域特定的实体词典
   - 优化实体向量的选择策略

3. 训练策略：
   - 使用学习率调度器
   - 实现早停机制
   - 使用梯度累积处理大批次

## 注意事项

1. 确保有足够的磁盘空间存储Wikipedia2Vec模型
2. 建议使用GPU进行训练
3. 根据实际GPU内存大小调整batch_size
4. 可以通过调整hidden_dim来控制模型大小
5. 首次运行时需要下载预训练模型，可能需要较长时间

## 后续优化方向

1. 支持更多类型的知识图谱嵌入
2. 添加更多的实体关系特征
3. 实现多GPU训练支持
4. 添加更多的评估指标
5. 优化内存使用效率
