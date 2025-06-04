import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Optional
from datasets.bert_processors.abstract_processor import BertProcessor, InputExample

class EntityEnhancedProcessor(BertProcessor):
    """处理带有实体向量的数据的处理器"""

    def __init__(self, entity_vector_path: str):
        super().__init__()
        self.entity_vectors = None
        if os.path.exists(entity_vector_path):
            try:
                if entity_vector_path.endswith('.npy'):
                    self.entity_vectors = np.load(entity_vector_path, allow_pickle=True)
                elif entity_vector_path.endswith('.json'):
                    with open(entity_vector_path, 'r', encoding='utf-8') as f:
                        self.entity_vectors = json.load(f)
                        self.entity_vectors = [np.array(vecs) for vecs in self.entity_vectors]
                else:
                    print(f"Unsupported entity vector file format: {entity_vector_path}")
            except Exception as e:
                print(f"Warning: Could not load entity vectors from {entity_vector_path}: {e}")
                print("Using zero vectors instead.")
        if self.entity_vectors is None:
            self.entity_vectors = []
        
    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """获取训练样本

        Args:
            data_dir: 数据目录

        Returns:
            examples: 训练样本列表
        """
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_train.json")),
            "train"
        )

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """获取验证样本

        Args:
            data_dir: 数据目录

        Returns:
            examples: 验证样本列表
        """
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_dev.json")),
            "dev"
        )

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        """获取测试样本

        Args:
            data_dir: 数据目录

        Returns:
            examples: 测试样本列表
        """
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_test.json")),
            "test"
        )

    def _read_json(self, input_file: str) -> List[Dict]:
        """读取JSON文件，兼容每行一个json对象的格式"""
        with open(input_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # 行式json
                return [json.loads(line) for line in f if line.strip()]

    def _create_examples(self, data: List[Dict], set_type: str) -> List[InputExample]:
        """创建输入样本

        Args:
            data: 数据列表
            set_type: 数据集类型(train/dev/test)

        Returns:
            examples: 输入样本列表
        """
        examples = []
        for i, item in enumerate(data):
            # 获取文本内容
            text = item["title"] + " " + item["abstract"]
            
            # 获取实体信息
            entities = item.get("entities", [])
            entity_vectors = item.get("entity_vectors", [])
            
            # 创建样本
            example = InputExample(
                guid=f"{set_type}-{i}",
                text_a=text,
                label=item.get("label", None),
                entity_vectors=np.array(entity_vectors) if entity_vectors else None
            )
            examples.append(example)
            
        return examples
        
    def convert_examples_to_features(self, examples: List[InputExample], max_seq_length: int, tokenizer) -> List[Dict]:
        """将样本转换为特征

        Args:
            examples: 样本列表
            max_seq_length: 最大序列长度
            tokenizer: 分词器

        Returns:
            features: 特征列表
        """
        features = []
        for example in examples:
            tokens = tokenizer.tokenize(example.text_a)
            if len(tokens) > max_seq_length - 2:  # 考虑[CLS]和[SEP]
                tokens = tokens[:(max_seq_length - 2)]
                
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # 填充
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            attention_mask = [1] * len(tokens) + [0] * len(padding)
            
            features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': example.label,
                'entity_vectors': example.entity_vectors
            })
            
        return features
        
    def get_data_loader(self, features: List[Dict], batch_size: int, shuffle: bool = False) -> DataLoader:
        """创建数据加载器

        Args:
            features: 特征列表
            batch_size: 批次大小
            shuffle: 是否打乱数据

        Returns:
            data_loader: 数据加载器
        """
        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
        
        # 处理实体向量
        has_entity_vectors = any(isinstance(f['entity_vectors'], np.ndarray) and f['entity_vectors'].size > 0 for f in features)
        if has_entity_vectors:
            max_entities = max(f['entity_vectors'].shape[0] for f in features if isinstance(f['entity_vectors'], np.ndarray))
            all_entity_vectors = torch.zeros(len(features), max_entities, 300)
            for i, f in enumerate(features):
                if isinstance(f['entity_vectors'], np.ndarray) and f['entity_vectors'].size > 0:
                    num_entities = f['entity_vectors'].shape[0]
                    all_entity_vectors[i, :num_entities] = torch.tensor(f['entity_vectors'])
        else:
            all_entity_vectors = torch.zeros(len(features), 1, 300)
        
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_entity_vectors, all_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
