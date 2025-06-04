import os
import json
import bz2
import pickle
from wikipedia2vec import Wikipedia2Vec
import numpy as np
import argparse

def load_wiki2vec_model(model_path):
    print(f"加载 Wikipedia2Vec 模型: {model_path}")
    return Wikipedia2Vec.load(model_path)

def get_entity_vector(wiki2vec, entity, dim=300):
    try:
        # 先查实体
        return wiki2vec.get_entity_vector(entity)
    except KeyError:
        try:
            # 查词向量
            return wiki2vec.get_word_vector(entity)
        except KeyError:
            # 没有则返回全零
            return np.zeros(dim, dtype=np.float32)

def process_dataset(input_file, output_file, wiki2vec_model_path):
    wiki2vec = load_wiki2vec_model(wiki2vec_model_path)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        entities = item.get('entities', [])
        vectors = []
        for entity_info in entities:
            # entity_info 可能是 (实体, 类型) 或 str
            if isinstance(entity_info, (list, tuple)):
                entity = entity_info[0]
            else:
                entity = entity_info
            vec = get_entity_vector(wiki2vec, entity)
            vectors.append(vec.tolist())
        item['entity_vectors'] = vectors
        if (i+1) % 50 == 0:
            print(f"已处理 {i+1} 条样本")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"处理完成，结果已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为实体获取Wikipedia2Vec向量')
    parser.add_argument('--input_file', type=str, default='data/exaapd/train_with_entities.json')
    parser.add_argument('--output_file', type=str, default='data/exaapd/train_with_vectors.json')
    parser.add_argument('--wiki2vec_model', type=str, default='models/wiki2vec/enwiki_20180420_300d.pkl')
    args = parser.parse_args()
    process_dataset(args.input_file, args.output_file, args.wiki2vec_model) 