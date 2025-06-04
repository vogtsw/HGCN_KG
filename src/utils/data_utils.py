import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        
        # 加载数据
        if mode == "train":
            data_file = os.path.join(args.data_dir, "train_kg.json")
        else:
            data_file = os.path.join(args.data_dir, f"{mode}.json")
            
        self.examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if '\t' in line:
                        _, json_str = line.strip().split('\t', 1)
                    else:
                        json_str = line.strip()
                    example = json.loads(json_str)
                    self.examples.append(example)
                except Exception as e:
                    print(f"处理数据时出错: {e}")
                    continue
                
        if mode == "train" and args.max_train_samples is not None:
            self.examples = self.examples[:args.max_train_samples]
            
        print(f"加载了 {len(self.examples)} 条{mode}数据")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 处理文本
        text = example.get('text', '')
        if not text and 'title' in example:
            text = example['title']
            
        encoding = self.tokenizer(
            text,
            max_length=self.args.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签
        label = torch.tensor(example.get('label', [0] * self.args.num_labels), dtype=torch.float)
        
        # 处理实体向量
        entity_vectors = example.get('entity_vectors', [])
        if entity_vectors:
            # 确保实体向量数量不超过最大值
            max_entities = min(len(entity_vectors), 50)  # 设置最大实体数为50
            entity_vectors = entity_vectors[:max_entities]
            # 如果实体数量不足，用零向量填充
            if len(entity_vectors) < 50:
                padding = [[0] * 300] * (50 - len(entity_vectors))
                entity_vectors.extend(padding)
        else:
            entity_vectors = [[0] * 300] * 50
            
        entity_vectors = torch.tensor(entity_vectors, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': label,
            'entity_vectors': entity_vectors
        }

def load_and_cache_examples(args, tokenizer, mode="train"):
    """加载并缓存数据集"""
    dataset = TextDataset(args, tokenizer, mode)
    return dataset 