import os
import json
import logging
import torch
import argparse
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

class DataProcessor:
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()

class ExAAPDProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_kg.json")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_dev.json")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_test.json")), "test")
    def get_labels(self):
        return list(range(2))
    def _read_json(self, input_file):
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
            # 合并所有文本字段
            text_parts = []
            if 'title' in line:
                text_parts.append(line['title'])
            if 'abstract' in line:
                text_parts.append(line['abstract'])
            if 'Introduction' in line:
                text_parts.append(line['Introduction'])
            if 'Related Work' in line:
                text_parts.append(line['Related Work'])
            if 'Conclusion' in line:
                text_parts.append(line['Conclusion'])
            
            text = ' '.join(text_parts)
            label = line.get('label', 0)
            entity_vectors = line.get('entity_vectors', None)
            examples.append(InputExample(guid=guid, text=text, label=label, entity_vectors=entity_vectors))
        return examples

class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity_vectors=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity_vectors = entity_vectors

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for ex in examples:
        tokens = tokenizer.tokenize(ex.text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        label_id = int(ex.label) if ex.label is not None else 0
        entity_vectors = ex.entity_vectors if ex.entity_vectors is not None else None
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id, entity_vectors))
    return features

def evaluate(model, processor, tokenizer, args, split, device):
    logger.info(f"加载{split}数据集...")
    examples = processor.get_dev_examples(args.data_dir) if split == 'dev' else processor.get_test_examples(args.data_dir)
    if not examples:
        logger.warning(f"{split}数据集为空，跳过评估")
        return
    logger.info(f"处理{split}数据特征...")
    features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
    if not features:
        logger.warning(f"{split}特征为空，跳过评估")
        return
    logger.info(f"准备{split}数据张量...")
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # entity_vectors 可能为None
    if hasattr(features[0], 'entity_vectors') and features[0].entity_vectors is not None:
        entity_vectors = torch.tensor([f.entity_vectors for f in features], dtype=torch.float)
    else:
        entity_vectors = None
    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids) if entity_vectors is None else TensorDataset(input_ids, input_mask, segment_ids, label_ids, entity_vectors)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        if entity_vectors is not None:
            input_ids, input_mask, segment_ids, label_ids, entity_vectors_batch = batch
            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, entity_vectors_batch)
        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask)
        preds = torch.argmax(outputs, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_ids.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    logger.info(f"{split}集评估结果：Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f"{split}集评估结果：Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

def test_model(args):
    logger.info("开始测试...")
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    processor = ExAAPDProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info(f"标签数量: {num_labels}")
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    config = BertConfig.from_pretrained(args.pretrained_model)
    model = ClassifyModel(config, num_labels, args.hidden_dim, args.dropout, args.pretrained_model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"已加载模型: {args.model_path}")
    for split in ['dev', 'test']:
        try:
            evaluate(model, processor, tokenizer, args, split, device)
        except Exception as e:
            logger.error(f"{split}评估过程中出错: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--pretrained_model", type=str, default="allenai/scibert_scivocab_uncased",
                        help="预训练模型名称或路径")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="评估批次大小")
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")
    parser.add_argument("--num_labels", type=int, default=2,
                        help="标签数量")
    parser.add_argument("--use_gpu", action="store_true",
                        help="是否使用GPU")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型路径")
    args = parser.parse_args()
    test_model(args)

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
        text_features = self.text_linear(text_features)
        text_features = F.relu(text_features)
        if kg_vectors.dim() == 1:
            kg_vectors = kg_vectors.unsqueeze(0)  # [1, 300]
        kg_features = self.kg_linear(kg_vectors)  # [num_entities, output_dim]
        kg_features = F.relu(kg_features)
        kg_attention = torch.sigmoid(self.kg_attention(kg_features))  # [num_entities, 1]
        kg_features = kg_features * kg_attention  # [num_entities, output_dim]
        kg_features_pooled = kg_features.mean(dim=0, keepdim=True)  # [1, output_dim]
        kg_features_pooled = kg_features_pooled.expand(text_features.size(0), -1)  # [batch_size, output_dim]
        combined_features = torch.cat([text_features, kg_features_pooled], dim=-1)  # [batch_size, output_dim*2]
        fused_features = self.fusion_linear(combined_features)
        fused_features = F.relu(fused_features)
        fused_features = self.dropout(fused_features)
        fused_features = self.layer_norm(fused_features)
        return fused_features

class ClassifyModel(nn.Module):
    def __init__(self, config, num_labels, hidden_dim, dropout, pretrained_model_name_or_path):
        super(ClassifyModel, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.kg_pooling = DecoupledGraphPooling(
            input_dim=768,  # BERT的隐藏维度
            output_dim=hidden_dim,
            dropout=dropout
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, entity_vectors=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS]标记的输出
        if entity_vectors is not None:
            if isinstance(entity_vectors, list):
                kg_vectors = torch.tensor(entity_vectors, dtype=torch.float32).to(input_ids.device)
            else:
                kg_vectors = entity_vectors
            features = self.kg_pooling(text_features, kg_vectors)
        else:
            features = text_features
        logits = self.classifier(features)
        return logits

if __name__ == "__main__":
    main() 