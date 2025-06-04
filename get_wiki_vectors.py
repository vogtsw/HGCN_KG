import json
import numpy as np
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec
import os

def load_entities(file_path: str):
    """加载实体数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_entity_vectors(entities, wiki2vec):
    """获取实体的Wikipedia2Vec向量"""
    vectors = []
    for doc_entities in tqdm(entities):
        doc_vectors = []
        for entity in doc_entities:
            try:
                # 尝试获取实体向量
                vector = wiki2vec.get_entity_vector(entity["text"])
                doc_vectors.append(vector.tolist())  # 转换为Python列表
            except KeyError:
                # 如果实体不存在,使用零向量
                doc_vectors.append([0.0] * 300)
        
        # 如果文档没有实体,添加一个零向量
        if not doc_vectors:
            doc_vectors.append([0.0] * 300)
            
        vectors.append(doc_vectors)
    
    return vectors

def main():
    # 设置输入输出目录
    input_dir = "data/exaapd_processed"
    output_dir = "data/exaapd_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载Wikipedia2Vec模型
    print("Loading Wikipedia2Vec model...")
    model_path = "models/wiki2vec/enwiki_20180420_300d.pkl"  # 修改为实际的模型路径
    wiki2vec = Wikipedia2Vec.load(model_path)
    
    # 加载实体
    print("Loading entities...")
    entities = load_entities(os.path.join(input_dir, "extracted_entities.json"))
    
    # 获取实体向量
    print("Getting entity vectors...")
    vectors = get_entity_vectors(entities, wiki2vec)
    
    # 保存向量
    print("Saving vectors...")
    with open(os.path.join(output_dir, "entity_vectors.json"), "w", encoding="utf-8") as f:
        json.dump(vectors, f)
    
    print("Done!")

if __name__ == "__main__":
    main() 