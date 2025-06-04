import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from wikipedia2vec import Wikipedia2Vec
import spacy
import argparse
from pathlib import Path

def extract_entities(text, nlp):
    """使用spaCy提取文本中的实体"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'WORK_OF_ART', 'LAW', 'PRODUCT']:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    return entities

def get_entity_vectors(entities, wiki2vec_model):
    """使用Wikipedia2Vec获取实体的向量表示"""
    entity_vectors = []
    for entity in entities:
        try:
            # 尝试获取实体的向量表示
            vector = wiki2vec_model.get_entity_vector(entity['text'])
            entity_vectors.append(vector)
        except KeyError:
            # 如果实体不在Wikipedia2Vec中，使用零向量
            entity_vectors.append(np.zeros(300))
    return np.array(entity_vectors)

def process_file(input_file, output_file, nlp, wiki2vec_model, max_seq_length=128):
    """处理单个文件，提取实体并获取向量表示"""
    print(f"Processing file: {input_file}")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 尝试读取第一行来检查格式
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            if first_line.startswith('['):
                # 标准JSON数组格式
                data = json.load(f)
            else:
                # 可能是JSONL格式（每行一个JSON对象）
                data = []
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {e}")
                        continue
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        raise
    
    print(f"Found {len(data)} items in {input_file}")
    
    processed_data = []
    for i, item in enumerate(tqdm(data, desc=f"Processing {input_file}")):
        try:
            if 'text' not in item:
                print(f"Warning: Item {i} has no 'text' field")
                continue
                
            text = item['text']
            entities = extract_entities(text, nlp)
            entity_vectors = get_entity_vectors(entities, wiki2vec_model)
            
            # 将实体和向量添加到原始数据中
            item['entities'] = entities
            item['entity_vectors'] = entity_vectors.tolist()
            processed_data.append(item)
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed_data)} items")
    
    # 保存处理后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"Saved processed data to {output_file}")
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--wiki2vec_model_path', type=str, required=True, help='Wikipedia2Vec模型路径')
    parser.add_argument('--max_seq_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    args = parser.parse_args()

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # 检查Wikipedia2Vec模型文件
    if not os.path.exists(args.wiki2vec_model_path):
        raise FileNotFoundError(f"Wikipedia2Vec model not found: {args.wiki2vec_model_path}")

    # 加载spaCy模型
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        raise
    
    # 加载Wikipedia2Vec模型
    print("Loading Wikipedia2Vec model...")
    try:
        wiki2vec_model = Wikipedia2Vec.load(args.wiki2vec_model_path)
    except Exception as e:
        print(f"Error loading Wikipedia2Vec model: {e}")
        raise
    
    # 处理训练集
    print("Processing training data...")
    train_input = os.path.join(args.data_dir, 'exAAPD_train.json')
    train_output = os.path.join(args.data_dir, 'exAAPD_train_with_entities.json')
    process_file(train_input, train_output, nlp, wiki2vec_model, args.max_seq_length)
    
    # 处理验证集
    print("Processing validation data...")
    dev_input = os.path.join(args.data_dir, 'exAAPD_dev.json')
    dev_output = os.path.join(args.data_dir, 'exAAPD_dev_with_entities.json')
    process_file(dev_input, dev_output, nlp, wiki2vec_model, args.max_seq_length)
    
    # 处理测试集
    print("Processing test data...")
    test_input = os.path.join(args.data_dir, 'exAAPD_test.json')
    test_output = os.path.join(args.data_dir, 'exAAPD_test_with_entities.json')
    process_file(test_input, test_output, nlp, wiki2vec_model, args.max_seq_length)
    
if __name__ == '__main__':
    main()
