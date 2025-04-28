# HGCN 更新日志

本文档详细说明了与原始HGCN代码相比的所有重要更新和改进。

## 主要更新

### 1. 知识增强
- **新增**: 集成了Wikipedia2Vec实体向量（300维）
- **新增**: 添加了实体提取和向量获取功能
- **改进**: 通过知识图谱增强了文本表示

### 2. 模型架构

#### 新增组件
- **EntityEnhancedHGCN**: 支持实体向量的新模型架构
- **注意力机制**: 用于融合文本和实体信息
- **BERT集成**: 使用BERT作为文本编码器

#### 原有组件的改进
- **数据处理器**: 扩展支持实体向量
- **批处理**: 优化了变长实体序列的处理
- **模型输入**: 支持多模态输入（文本+实体）

### 3. 文件结构变化

#### 新增文件
```
HGCN/
├── datasets/bert_processors/
│   └── entity_enhanced_processor.py    # 新增
├── models/
│   └── entity_enhanced_hgcn.py        # 新增
├── src/
│   ├── download_wiki2vec.py           # 新增
│   └── process_entity_vectors.py      # 新增
└── prepare_training_data.py           # 新增
```

#### 修改的文件
- `requirements.txt`: 添加了新的依赖
- `train.py`: 增加了对实体向量的支持

### 4. 功能增强

#### 数据处理
- **原有**: 仅支持文本输入
- **现在**: 
  - 支持实体提取
  - 支持实体向量获取
  - 支持多模态数据融合

#### 模型能力
- **原有**: 仅处理文本特征
- **现在**:
  - 处理文本和实体特征
  - 通过注意力机制动态融合特征
  - 支持知识增强的文本表示

#### 训练流程
- **原有**: 单一模态训练
- **现在**:
  - 支持多模态训练
  - 优化了批处理机制
  - 添加了更多评估指标

### 5. 性能改进

#### 模型性能
- 通过知识图谱增强提升了文本理解能力
- 注意力机制提供了更好的特征融合
- BERT编码器提供了更强的文本表示

#### 计算效率
- 优化了批处理机制
- 改进了数据加载效率
- 支持GPU加速

### 6. API变化

#### 新增API
- `EntityEnhancedProcessor.process_entities()`
- `EntityEnhancedHGCN.forward()`
- `process_entity_vectors.extract_entities()`

#### 修改的API
- `train()`: 添加了实体向量支持
- `evaluate()`: 扩展了评估指标
- `load_data()`: 支持多模态数据

### 7. 配置变更

#### 新增配置项
- `wiki2vec_model_path`: Wikipedia2Vec模型路径
- `entity_dim`: 实体向量维度
- `attention_heads`: 注意力头数量

#### 修改的配置项
- `hidden_dim`: 支持更大的维度
- `batch_size`: 考虑实体数量的动态调整
- `learning_rate`: 针对多模态学习优化

## 不兼容变更

1. 数据格式
   - 训练数据需要包含实体信息
   - 模型输入格式发生变化

2. 模型接口
   - forward方法参数变更
   - 配置文件格式更新

3. 训练脚本
   - 需要先运行实体处理
   - 参数列表有所变化

## 迁移指南

### 从旧版本升级
1. 安装新的依赖：
```bash
pip install -r requirements.txt
```

2. 处理数据：
```bash
python src/process_entity_vectors.py
python prepare_training_data.py
```

3. 更新训练脚本：
```bash
python train_entity_enhanced.py  # 替代原来的train.py
```

### 数据迁移
1. 为现有数据添加实体标注
2. 生成实体向量
3. 更新数据格式

## 后续规划

### 计划中的功能
1. 支持更多知识图谱
2. 添加更多实体关系特征
3. 实现多GPU训练
4. 优化内存使用
5. 添加更多评估指标

### 已知问题
1. 大规模数据处理效率待优化
2. 实体提取质量依赖于NER性能
3. 内存占用较大

## 参考文献
1. Wikipedia2Vec论文
2. BERT论文
3. 原始HGCN论文
