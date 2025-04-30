import os
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm

def load_entities_and_vectors(entity_file: str, vector_file: str) -> tuple:
    """加载实体和向量数据
    
    Args:
        entity_file: 实体文件路径
        vector_file: 向量文件路径
        
    Returns:
        entities: 实体列表
        vectors: 向量数组
    """
    with open(entity_file, "r", encoding="utf-8") as f:
        entities = json.load(f)
    vectors = np.load(vector_file, allow_pickle=True)
    return entities, vectors

def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON格式的数据文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        data: 数据列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # 找到第一个JSON对象的开始
        start_idx = content.find('{')
        if start_idx == -1:
            raise ValueError(f"No JSON content found in {file_path}")
        content = content[start_idx:]
        
        # 尝试解析单个JSON对象
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取所有JSON对象
            data = []
            current_pos = 0
            while True:
                try:
                    obj_start = content.find('{', current_pos)
                    if obj_start == -1:
                        break
                    
                    decoder = json.JSONDecoder()
                    obj, end = decoder.raw_decode(content[obj_start:])
                    data.append(obj)
                    current_pos = obj_start + end
                except json.JSONDecodeError:
                    if current_pos >= len(content):
                        break
                    current_pos += 1
            
            if not data:
                raise ValueError(f"No valid JSON objects found in {file_path}")
            return data

def process_document(doc: Dict, doc_entities: List[Dict]) -> Dict:
    """处理单个文档
    
    Args:
        doc: 文档字典
        doc_entities: 文档中的实体列表
        
    Returns:
        processed_doc: 处理后的文档字典
    """
    # 复制原始文档数据
    processed_doc = doc.copy()
    
    # 添加实体索引
    processed_doc["entity_indices"] = []
    
    # 记录每个实体的位置
    for entity in doc_entities:
        processed_doc["entity_indices"].append({
            "text": entity["text"],
            "type": entity["type"],
            "vector_index": len(processed_doc["entity_indices"])
        })
    
    return processed_doc

def prepare_dataset(input_dir: str, output_dir: str):
    """准备训练数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载实体和向量数据
    entities, vectors = load_entities_and_vectors(
        os.path.join(input_dir, "extracted_entities.json"),
        os.path.join(input_dir, "entity_vectors.npy")
    )
    
    # 处理训练集
    print("Processing training set...")
    train_data = load_json_data(os.path.join(input_dir, "exAAPD_train.json"))
    
    processed_train = []
    for i, doc in enumerate(tqdm(train_data)):
        processed_doc = process_document(doc, entities[i] if i < len(entities) else [])
        processed_train.append(processed_doc)
    
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(processed_train, f, indent=2, ensure_ascii=False)
    
    # 处理测试集
    print("Processing test set...")
    test_data = load_json_data(os.path.join(input_dir, "exAAPD_test.json"))
    
    processed_test = []
    for doc in tqdm(test_data):
        processed_doc = process_document(doc, [])  # 测试集暂时不添加实体
        processed_test.append(processed_doc)
    
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(processed_test, f, indent=2, ensure_ascii=False)
    
    # 处理验证集（使用测试集）
    print("Processing dev set...")
    with open(os.path.join(output_dir, "dev.json"), "w", encoding="utf-8") as f:
        json.dump(processed_test, f, indent=2, ensure_ascii=False)
    
    # 复制实体向量
    print("Copying entity vectors...")
    np.save(os.path.join(output_dir, "entity_vectors.npy"), vectors)
    
    print("Done!")

def main():
    # 设置输入输出目录
    input_dir = "data/exaapd"
    output_dir = "data/exaapd_processed"
    
    # 准备数据集
    prepare_dataset(input_dir, output_dir)

if __name__ == "__main__":
    main()
