import logging
import torch
from torch.utils.data import TensorDataset
import numpy as np
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity_vectors=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_vectors = entity_vectors

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        
        # 截断序列
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        
        # 添加[CLS]和[SEP]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 创建mask
        input_mask = [1] * len(input_ids)
        
        # padding
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        # 处理标签
        if example.label is not None:
            if isinstance(example.label, list):
                label_id = example.label
            else:
                label_id = [0] * 54  # 假设有54个标签
                label_id[example.label] = 1
        else:
            label_id = None
            
        # 处理实体向量
        entity_vectors = None
        if hasattr(example, 'entity_vectors') and example.entity_vectors:
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

tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = BertModel.from_pretrained("allenai/scibert_scivocab_uncased") 