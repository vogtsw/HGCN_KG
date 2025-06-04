import os
import json
import torch
import argparse
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class EntityExtractor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        bert_dir = "models/bert-base-uncased/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        base_model = BertForTokenClassification.from_pretrained(bert_dir, num_labels=9)
        
        # 加载训练好的模型权重
        state_dict = torch.load(model_path, map_location=device)
        # 只加载与BertForTokenClassification模型结构匹配的权重
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_key = 'bert.' + k[8:]
                if new_key in base_model.state_dict():
                    new_state_dict[new_key] = v  # 只加载匹配的键
        base_model.load_state_dict(new_state_dict, strict=False)  # 允许部分加载
        self.model = base_model
        self.model.to(device)
        self.model.eval()
        
        # 实体标签映射
        self.id2label = {
            0: "O",
            1: "B-PER", 2: "I-PER",  # 人名
            3: "B-ORG", 4: "I-ORG",  # 组织
            5: "B-MISC", 6: "I-MISC",  # 其他
            7: "B-LOC", 8: "I-LOC"  # 地点
        }
        
    def extract_entities(self, data):
        """从论文数据中提取实体"""
        all_entities = []
        
        # 处理标题
        if 'title' in data:
            title_entities = self._extract_entities_from_text(data['title'])
            all_entities.extend(title_entities)
        
        # 处理摘要
        if 'abstract' in data:
            abstract_entities = self._extract_entities_from_text(data['abstract'])
            all_entities.extend(abstract_entities)
        
        # 处理正文部分
        for key in ['Introduction', 'Related work', 'Conclusion']:
            if key in data:
                section_entities = self._extract_entities_from_text(data[key])
                all_entities.extend(section_entities)
        
        # 去重并返回
        unique_entities = {}
        for entity, type_ in all_entities:
            if entity not in unique_entities:
                unique_entities[entity] = type_
        
        return list(unique_entities.items())

    def _extract_entities_from_text(self, text):
        """从文本中提取实体"""
        # 对文本进行分词
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        # 获取预测结果
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
        
        # 解码预测结果
        entities = []
        current_entity = []
        current_type = None
        
        for i, pred in enumerate(predictions[0]):
            if pred != 0:  # 0是O标签
                token = self.tokenizer.decode([input_ids[0][i]])
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    label = self.id2label[pred.item()]
                    
                    if label.startswith('B-'):  # 开始新实体
                        if current_entity:  # 保存之前的实体
                            entity_text = ''.join(current_entity).replace('##', '')
                            if len(entity_text) > 1 and not entity_text.isdigit():  # 过滤掉太短的实体和纯数字
                                entities.append((entity_text, current_type))
                        current_entity = [token]
                        current_type = label[2:]
                    elif label.startswith('I-') and current_entity:  # 继续当前实体
                        current_entity.append(token)
        
        # 处理最后一个实体
        if current_entity:
            entity_text = ''.join(current_entity).replace('##', '')
            if len(entity_text) > 1 and not entity_text.isdigit():
                entities.append((entity_text, current_type))
        
        # 过滤掉一些常见的非实体词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        entities = [(text, type_) for text, type_ in entities if text.lower() not in stop_words]
        
        return entities

def process_dataset(input_file, output_file, extractor, max_samples=None):
    """处理数据集并提取实体
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        extractor: 实体提取器实例
        max_samples: 最大处理样本数，None表示处理所有样本
    """
    print(f"开始处理数据集: {input_file}")
    if max_samples:
        print(f"将处理最多 {max_samples} 条数据")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理每一行
    processed_data = []
    for i, line in enumerate(lines):
        if max_samples and i >= max_samples:
            break
            
        try:
            # 分割ID和JSON数据
            id_part, json_part = line.strip().split('\t', 1)
            
            # 解析JSON数据
            data = json.loads(json_part)
            
            # 提取实体
            entities = extractor.extract_entities(data)
            
            # 添加实体到数据中
            data['entities'] = entities
            processed_data.append(data)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 条数据")
                
        except Exception as e:
            print(f"处理第 {i + 1} 行时出错: {str(e)}")
            continue
    
    # 保存处理后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到: {output_file}")
    print(f"共处理 {len(processed_data)} 条数据")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--num_samples", type=int, default=None, help="处理的样本数量，默认处理所有数据")
    args = parser.parse_args()

    # 加载NER模型
    model = BertForTokenClassification.from_pretrained("checkpoints/entity_enhanced/best_model.pt")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.eval()

    # 读取输入数据
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.num_samples is not None and i >= args.num_samples:
                break
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效的JSON行: {e}")
                continue

    # 处理数据
    processed_data = []
    for item in tqdm(data, desc="处理数据"):
        text = item.get('title', '') + ' ' + item.get('abstract', '')
        entities = extract_entities(text, model, tokenizer)
        item['entities'] = entities
        processed_data.append(item)

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main() 