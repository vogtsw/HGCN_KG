import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from args import get_args

class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分词
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_length - 2:  # 考虑[CLS]和[SEP]
            tokens = tokens[:(self.max_length - 2)]
        
        # 添加特殊标记
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        
        # 处理标签
        if len(label) > self.max_length - 2:
            label = label[:(self.max_length - 2)]
        label = [0] + label + [0]  # 为[CLS]和[SEP]添加标签
        label += [0] * (self.max_length - len(label))  # Padding
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_json_data(file_path, max_samples=None):
    """加载JSON数据，处理可能的二进制前缀
    
    Args:
        file_path: JSON文件路径
        max_samples: 最大样本数，如果为None则加载所有样本
        
    Returns:
        data: JSON数据列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        start_idx = content.find('{')
        content = content[start_idx:]
        
        # 尝试加载JSON数据
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            # 如果是多个JSON对象，尝试逐行解析
            lines = content.strip().split('\n')
            data = []
            for line in lines:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    
    # 如果指定了最大样本数，只返回前max_samples个样本
    if max_samples is not None:
        data = data[:max_samples]
    
    return data

def prepare_data(data, tokenizer):
    """准备训练数据
    
    Args:
        data: 原始数据列表
        tokenizer: BERT分词器
        
    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    texts = []
    labels = []
    
    for doc in data:
        text = doc.get("text", "") or doc.get("title", "")
        if isinstance(text, str) and text.strip():
            # 分词
            tokens = tokenizer.tokenize(text)
            # 创建标签序列（默认为O标签）
            label = [0] * len(tokens)
            # 添加实体标签
            if "entities" in doc:
                for entity in doc["entities"]:
                    start = entity["start"]
                    end = entity["end"]
                    if start < len(label):
                        label[start] = 1  # B-tag
                        for i in range(start + 1, min(end + 1, len(label))):
                            label[i] = 2  # I-tag
            
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch
    
    Args:
        model: BERT模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        
    Returns:
        loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # 解析参数
    args = get_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载BERT分词器和模型
    print("Loading BERT model...")
    model_name = 'bert-base-cased'  # 使用预训练的cased模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=3)  # O, B-ENTITY, I-ENTITY
    model.to(device)
    
    # 加载训练数据
    print("Loading training data...")
    train_file = os.path.join(args.data_dir, "exAAPD_train.json")
    train_data = load_json_data(train_file)
    print(f"Loaded {len(train_data)} training samples")
    
    # 准备训练数据
    texts, labels = prepare_data(train_data, tokenizer)
    dataset = NERDataset(texts, labels, tokenizer, args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(dataloader) * 10  # 10个epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # 训练模型
    print("Training model...")
    for epoch in range(10):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    # 保存模型
    print("Saving model...")
    output_dir = os.path.join(args.data_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "ner_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "ner_model"))
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
