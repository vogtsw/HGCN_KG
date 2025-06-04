# coding=utf-8
import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import scipy.linalg
import json

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
import torch.nn as nn
from tqdm import tqdm, trange
from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from transformers import get_linear_schedule_with_warmup
from transformers import BertConfig, BertTokenizer, BertModel
from torch.optim import AdamW
from args import get_args
from datasets.bert_processors.aapd_processor import (
    exAAPDProcessor_has_structure, exPFDProcessor_has_structure,
    exLitCovidProcessor_has_structure, exMeSHProcessor_has_structure,
    exAAPDProcessor_no_structure, exPFDProcessor_no_structure,
    exLitCovidProcessor_no_structure, exMeSHProcessor_no_structure
)
from common.constants import *
import copy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample:
    def __init__(self, guid, text, label=None, entity_vectors=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.entity_vectors = entity_vectors

class ExAAPDProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """获取训练集样本"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_kg.json")), "train")

    def get_dev_examples(self, data_dir):
        """获取验证集样本"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """获取测试集样本"""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_test.json")), "test")

    def _read_json(self, input_file):
        """读取JSON文件"""
        data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"跳过无效的JSON行: {e}")
                        continue
            logger.info(f"成功读取 {len(data)} 条数据")
        except Exception as e:
            logger.error(f"读取文件 {input_file} 时出错: {str(e)}")
        return data

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = line.get('id', str(i))
            text = line.get('title', '') + ' ' + line.get('abstract', '')
            label = line.get('label', 0)
            entity_vectors = line.get('entity_vectors', None)
            examples.append(InputExample(guid=guid, text=text, label=label, entity_vectors=entity_vectors))
        return examples

def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

class DecoupledGraphPooling(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(DecoupledGraphPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 文本特征处理
        self.text_linear = nn.Linear(input_dim, output_dim)
        
        # KG向量处理
        self.kg_linear = nn.Linear(300, output_dim)  # 300是Wikipedia2Vec的维度
        self.kg_attention = nn.Linear(output_dim, 1)
        
        # 特征融合
        self.fusion_linear = nn.Linear(output_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, text_features, kg_vectors):
        # 处理文本特征
        text_features = self.text_linear(text_features)
        text_features = F.relu(text_features)
        
        # 处理KG向量
        if kg_vectors.dim() == 1:
            kg_vectors = kg_vectors.unsqueeze(0)  # [1, 300]
        kg_features = self.kg_linear(kg_vectors)  # [num_entities, output_dim]
        kg_features = F.relu(kg_features)
        
        # 计算注意力权重
        kg_attention = torch.sigmoid(self.kg_attention(kg_features))  # [num_entities, 1]
        kg_features = kg_features * kg_attention  # [num_entities, output_dim]
        
        # 对所有实体做mean池化，得到[1, output_dim]
        kg_features_pooled = kg_features.mean(dim=0, keepdim=True)  # [1, output_dim]
        # 扩展到batch_size
        kg_features_pooled = kg_features_pooled.expand(text_features.size(0), -1)  # [batch_size, output_dim]
        
        # 特征融合
        combined_features = torch.cat([text_features, kg_features_pooled], dim=-1)  # [batch_size, output_dim*2]
        fused_features = self.fusion_linear(combined_features)
        fused_features = F.relu(fused_features)
        
        # 应用dropout和layer normalization
        fused_features = self.dropout(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        return fused_features

class ClassifyModel(nn.Module):
    def __init__(self, config, num_labels, hidden_dim, dropout, pretrained_model_name_or_path):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # 添加KG向量处理层
        self.kg_pooling = DecoupledGraphPooling(
            input_dim=768,  # BERT的隐藏维度
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, entity_vectors=None):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS]标记的输出
        
        # 处理KG向量
        if entity_vectors is not None:
            # 将实体向量列表转换为张量
            if isinstance(entity_vectors, list):
                # 如果entity_vectors是列表，将其转换为张量
                kg_vectors = torch.tensor(entity_vectors, dtype=torch.float32).to(input_ids.device)
            else:
                # 如果已经是张量，直接使用
                kg_vectors = entity_vectors
            # 通过KG池化层
            features = self.kg_pooling(text_features, kg_vectors)
        else:
            features = text_features
        
        # 分类
        logits = self.classifier(features)
        
        return logits

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity_vectors=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_vectors = entity_vectors

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for example in examples:
        # 对文本进行分词
        tokens = tokenizer.tokenize(example.text)
        
        # 截断序列
        if len(tokens) > max_seq_length - 2:  # 为[CLS]和[SEP]预留位置
            tokens = tokens[:(max_seq_length - 2)]
        
        # 添加特殊标记
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 创建input mask
        input_mask = [1] * len(input_ids)
        
        # 填充
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = [0] * max_seq_length
        
        # 确保长度正确
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        # 处理标签
        label_id = example.label
        
        # 处理实体向量
        entity_vectors = example.entity_vectors
        
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                entity_vectors=entity_vectors
            )
        )
    return features

def test_model(args, model_path):
    """测试模型"""
    logger.info("开始测试...")
    
    # 加载处理器
    processor = ExAAPDProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(f"标签数量: {num_labels}")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    # 准备模型
    config = BertConfig.from_pretrained(args.pretrained_model)
    model = ClassifyModel(config, num_labels, args.hidden_dim, args.dropout, args.pretrained_model)
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    logger.info(f"已加载模型: {model_path}")
    
    # 测试dev和test数据集
    for split in ['dev', 'test']:
        try:
            logger.info(f"加载{split}数据集...")
            examples = processor.get_dev_examples(args.data_dir) if split == 'dev' else processor.get_test_examples(args.data_dir)
            if not examples:
                logger.warning(f"{split}数据集为空，跳过评估")
                continue
            
            logger.info(f"处理{split}数据特征...")
            features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
            if not features:
                logger.warning(f"{split}特征为空，跳过评估")
                continue
            
            logger.info(f"准备{split}数据张量...")
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(args.device)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(args.device)
            label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(args.device)
            
            logger.info(f"开始{split}评估...")
            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, features[0].entity_vectors if features[0].entity_vectors is not None else None)
                predictions = torch.argmax(outputs, dim=-1)
                accuracy = (predictions == label_ids).float().mean()
                logger.info(f"{split} 准确率: {accuracy:.4f}")
                
                # 计算详细指标
                true_positives = ((predictions == 1) & (label_ids == 1)).sum().float()
                false_positives = ((predictions == 1) & (label_ids == 0)).sum().float()
                false_negatives = ((predictions == 0) & (label_ids == 1)).sum().float()
                
                precision = true_positives / (true_positives + false_positives + 1e-10)
                recall = true_positives / (true_positives + false_negatives + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                
                logger.info(f"{split} 详细指标:")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1 Score: {f1:.4f}")
                
        except Exception as e:
            logger.error(f"{split}评估过程中出错: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="输入数据目录")
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--pretrained_model", default="allenai/scibert_scivocab_uncased", type=str,
                        help="预训练模型名称或路径")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="最大序列长度")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="评估批次大小")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="学习率")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="训练轮数")
    parser.add_argument("--hidden_dim", default=768, type=int,
                        help="隐藏层维度")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout率")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="标签数量")
    parser.add_argument("--use_gpu", action="store_true",
                        help="是否使用GPU")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    parser.add_argument("--do_train", action="store_true",
                        help="是否训练")
    parser.add_argument("--do_eval", action="store_true",
                        help="是否评估")
    parser.add_argument("--model_path", type=str, default=None,
                        help="训练好的模型路径")
    parser.add_argument("--do_test", action="store_true",
                        help="是否进行测试")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(args)

    # 加载处理器
    processor = ExAAPDProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(f"标签数量: {num_labels}")
    
    # 加载tokenizer
    logger.info(f"加载模型: {args.pretrained_model}")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    # 准备模型
    config = BertConfig.from_pretrained(args.pretrained_model)
    model = ClassifyModel(config, num_labels, args.hidden_dim, args.dropout, args.pretrained_model)
    model.to(args.device)
    logger.info("模型已加载到设备")

    # 准备优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 准备学习率调度器
    logger.info("开始加载训练数据...")
    train_examples = processor.get_train_examples(args.data_dir)
    logger.info(f"训练样本数: {len(train_examples)}")
    num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    # 训练
    if args.do_train:
        logger.info("开始训练...")
        model.train()
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f"Epoch {epoch + 1}/{int(args.num_train_epochs)}")
            for step, example in enumerate(train_examples):
                # 处理输入
                features = convert_examples_to_features([example], args.max_seq_length, tokenizer)
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
                input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(args.device)
                segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(args.device)
                label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(args.device)
                
                # 前向传播
                outputs = model(input_ids, segment_ids, input_mask, features[0].entity_vectors)
                loss = nn.CrossEntropyLoss()(outputs, label_ids)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if step % 100 == 0:
                    logger.info(f"Step {step}, Loss: {loss.item():.4f}")
        
        # 保存模型
        output_dir = os.path.join(args.data_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        logger.info(f"模型已保存到: {output_dir}")
        
    # 评估
    if args.do_eval:
        logger.info("开始评估...")
        model.eval()
        for split in ['dev', 'test']:
            try:
                logger.info(f"加载{split}数据集...")
                examples = processor.get_dev_examples(args.data_dir) if split == 'dev' else processor.get_test_examples(args.data_dir)
                if not examples:
                    logger.warning(f"{split}数据集为空，跳过评估")
                    continue
                
                logger.info(f"处理{split}数据特征...")
                features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
                if not features:
                    logger.warning(f"{split}特征为空，跳过评估")
                    continue
                
                logger.info(f"准备{split}数据张量...")
                input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(args.device)
                input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(args.device)
                segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(args.device)
                label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(args.device)
                
                logger.info(f"开始{split}评估...")
                with torch.no_grad():
                    outputs = model(input_ids, segment_ids, input_mask, features[0].entity_vectors if features[0].entity_vectors is not None else None)
                    predictions = torch.argmax(outputs, dim=-1)
                    accuracy = (predictions == label_ids).float().mean()
                    logger.info(f"{split} 准确率: {accuracy:.4f}")
            except Exception as e:
                logger.error(f"{split}评估过程中出错: {str(e)}")
                continue

    if args.do_test:
        if args.model_path is None:
            args.model_path = os.path.join(args.data_dir, "models", "pytorch_model.bin")
        test_model(args, args.model_path)

if __name__ == "__main__":
    main() 