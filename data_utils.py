import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from wikipedia2vec import Wikipedia2Vec

class InputExample(object):
    def __init__(self, text, label, entities=None, entity_vectors=None):
        self.text = text
        self.label = label
        self.entities = entities
        self.entity_vectors = entity_vectors

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity_vectors=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_vectors = entity_vectors

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

class ExaapdProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_train.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_dev.json")), "dev")

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, data, set_type):
        examples = []
        for (i, item) in enumerate(data):
            text = item['text']
            label = item['label']
            entities = item.get('entities', [])
            entity_vectors = item.get('entity_vectors', [])
            examples.append(InputExample(text=text, label=label, entities=entities, entity_vectors=entity_vectors))
        return examples

class ExaapdDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_length, wiki2vec_model_path):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # 加载Wikipedia2Vec模型
        self.wiki2vec = Wikipedia2Vec.load(wiki2vec_model_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        features = convert_example_to_feature(example, self.tokenizer, self.max_seq_length, self.wiki2vec)
        return features

def convert_example_to_feature(example, tokenizer, max_seq_length, wiki2vec):
    # 处理文本
    tokens = tokenizer.tokenize(example.text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    # 处理实体向量
    entity_vectors = None
    if example.entities:
        try:
            # 获取每个实体的向量
            vectors = []
            for entity in example.entities:
                try:
                    # 尝试获取实体向量
                    vector = wiki2vec.get_entity_vector(entity)
                    vectors.append(vector)
                except KeyError:
                    # 如果实体不在词表中,使用零向量
                    vectors.append(np.zeros(300))
            # 将所有实体向量平均
            entity_vectors = np.mean(vectors, axis=0)
        except Exception as e:
            print(f"Error processing entity vectors: {e}")
            entity_vectors = np.zeros(300)
    
    # 填充
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
        entity_vectors=entity_vectors
    ) 