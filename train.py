import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from data_utils import ExaapdProcessor, ExaapdDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 加载数据
    processor = ExaapdProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    
    # 创建数据集
    train_dataset = ExaapdDataset(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        wiki2vec_model_path=args.wiki2vec_model_path
    )
    
    dev_dataset = ExaapdDataset(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        wiki2vec_model_path=args.wiki2vec_model_path
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True
    )
    
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model,
        num_labels=len(processor.get_labels())
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    best_acc = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # 将数据移到GPU
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            # 前向传播
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['input_mask'],
                token_type_ids=batch['segment_ids'],
                labels=batch['label_id'],
                entity_vectors=batch['entity_vectors']
            )
            
            loss = outputs[0]
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dev_dataloader:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['input_mask'],
                    token_type_ids=batch['segment_ids'],
                    entity_vectors=batch['entity_vectors']
                )
                
                logits = outputs[0]
                pred = torch.argmax(logits, dim=1)
                correct += (pred == batch['label_id']).sum().item()
                total += batch['label_id'].size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1}, Dev Accuracy: {acc:.4f}")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # 添加参数
    parser.add_argument("--data_dir", default="data/exaapd", type=str)
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
    parser.add_argument("--wiki2vec_model_path", default="models/wiki2vec/enwiki_20180420_300d.pkl", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    train(args) 