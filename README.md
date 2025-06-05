# HGCN with KG Enhancement

本项目基于HGCN（Hierarchical Graph Convolutional Network）模型，结合知识图谱（KG）增强文本表示。主要流程包括：使用BERT-NER提取实体，用Wikipedia2Vec获取实体向量，将KG向量融合到训练数据，并修改HGCN模型以支持KG增强。

## 环境准备

- Python 3.6+
- PyTorch 1.7+
- transformers
- wikipedia2vec
- 其他依赖见 `requirements.txt`

## 模型下载

### BERT-NER模型
- 下载地址：`checkpoints/entity_enhanced/best_model.pt`
- 说明：基于`bert-base-uncased`的NER模型，用于实体抽取。

### Wikipedia2Vec模型
- 下载地址：`models/wiki2vec/enwiki_20180420_300d.pkl`
- 说明：预训练的Wikipedia2Vec模型，用于获取实体向量。

### SciBERT模型
- 下载地址：`allenai/scibert_scivocab_uncased`（通过HuggingFace自动下载）
- 说明：用于文本编码和分类。

## 数据准备

### 初始数据 下载地址：https://drive.google.com/drive/folders/1g9s_UiaTVC0GK80s56tiR-Tby-Jbi-wy 修改下名字如下
- 原始数据集存放在 `data/exaapd/` 目录下，包括：
  - `train.json`：训练集
  - `dev.json`：验证集
  - `test.json`：测试集



## 运行流程

### 1. 提取实体（NER）
- 代码：`extract_entities.py`
- 输入：`data/exaapd/train.json`（或`sample_train.json`）
- 输出：`data/exaapd/train_with_entities.json`（或`sample_train_with_entities.json`）
- 命令：
  ```bash
  # 处理全部数据
  python extract_entities.py --input_file data/exaapd/train.json --output_file data/exaapd/train_with_entities.json
  
  ```

### 2. 获取实体向量（Wikipedia2Vec）
- 代码：`get_entity_vectors.py`
- 输入：`data/exaapd/train_with_entities.json`（或`sample_train_with_entities.json`）
- 输出：`data/exaapd/train_with_vectors.json`（或`sample_train_with_vectors.json`）
- 命令：
  ```bash
  # 处理全部数据
  python get_entity_vectors.py --input_file data/exaapd/train_with_entities.json --output_file data/exaapd/train_with_vectors.json
  

  ```

### 3. 融合KG向量到训练数据
- 代码：`merge_kg_vectors.py`
- 输入：`data/exaapd/train.json`（或`sample_train.json`）和`data/exaapd/train_with_vectors.json`（或`sample_train_with_vectors.json`）
- 输出：`data/exaapd/train_kg.json`（或`sample_train_kg.json`）
- 命令：
  ```bash
  # 处理全部数据
  python merge_kg_vectors.py --train_file data/exaapd/train.json --vectors_file data/exaapd/train_with_vectors.json --output_file data/exaapd/train_kg.json
   ```


### 4. 训练HGCN模型（支持KG增强）
- 代码：`src/train.py`
- 输入：`data/exaapd/train_kg.json`（或`sample_train_kg.json`）
- 输出：训练好的模型权重（如`checkpoints/kg_enhanced/pytorch_model.bin`）
- 命令：
  ```bash
  python src/train.py --data_dir data/exaapd --dataset exAAPD_hs --pretrained_model allenai/scibert_scivocab_uncased --max_seq_length 128 --train_batch_size 32 --eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --hidden_dim 768 --dropout 0.1 --num_labels 2 --use_gpu --seed 42 --do_train --do_eval
  ```

## 快速验证流程
如果要快速验证整个流程，可以按以下步骤处理100条数据：

```bash
# 1. 提取实体
python extract_entities.py --input_file data/exaapd/train.json --output_file data/exaapd/train_with_entities.json --num_samples 100

# 2. 获取实体向量
python get_entity_vectors.py --input_file data/exaapd/train_with_entities.json --output_file data/exaapd/train_with_vectors.json --num_samples 100

# 3. 融合KG向量
python merge_kg_vectors.py --train_file data/exaapd/train.json --vectors_file data/exaapd/train_with_vectors.json --output_file data/exaapd/train_kg.json --num_samples 100

# 4. 训练模型
python src/train.py --data_dir data/exaapd --dataset exAAPD_hs --pretrained_model allenai/scibert_scivocab_uncased --max_seq_length 128 --train_batch_size 32 --eval_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --hidden_dim 768 --dropout 0.1 --num_labels 2 --use_gpu --seed 42 --do_train --do_eval
```

## 注意事项
- 确保模型文件已下载到指定路径。
- 数据文件需为JSON或JSONL格式，每条数据包含`title`、`abstract`、`label`等字段。
- 若使用demo数据，请将命令中的文件名替换为`sample_*.json`。




