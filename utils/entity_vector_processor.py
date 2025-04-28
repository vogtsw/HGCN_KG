import os
import numpy as np
from wikipedia2vec import Wikipedia2Vec
from typing import List, Dict, Optional, Tuple

class EntityVectorProcessor:
    def __init__(self, wiki2vec_model_path: str, vector_dim: int = 300):
        """初始化EntityVectorProcessor
        
        Args:
            wiki2vec_model_path: Wikipedia2Vec模型文件路径
            vector_dim: 向量维度，默认300
        """
        self.vector_dim = vector_dim
        self.wiki2vec = Wikipedia2Vec.load(wiki2vec_model_path)
        
    def get_entity_vector(self, entity_text: str) -> np.ndarray:
        """获取单个实体的向量表示
        
        Args:
            entity_text: 实体文本
            
        Returns:
            entity_vector: 实体的向量表示
        """
        try:
            # 尝试直接获取实体向量
            entity_vector = self.wiki2vec.get_entity_vector(entity_text)
        except KeyError:
            try:
                # 如果失败，尝试使用word vectors的平均值
                words = entity_text.split()
                word_vectors = [self.wiki2vec.get_word_vector(word) for word in words]
                entity_vector = np.mean(word_vectors, axis=0)
            except KeyError:
                # 如果还是失败，返回零向量
                entity_vector = np.zeros(self.vector_dim)
        
        return entity_vector
    
    def get_entity_vectors(self, entities: List[Dict]) -> Dict[str, np.ndarray]:
        """获取多个实体的向量表示
        
        Args:
            entities: 实体列表，每个实体是一个字典，包含type和text字段
            
        Returns:
            entity_vectors: 实体向量字典，key是实体文本，value是向量
        """
        entity_vectors = {}
        for entity in entities:
            entity_text = entity['text']
            if entity_text not in entity_vectors:
                entity_vectors[entity_text] = self.get_entity_vector(entity_text)
        return entity_vectors
    
    def get_document_entity_vectors(self, document_entities: List[List[Dict]]) -> List[Dict[str, np.ndarray]]:
        """获取文档列表中所有实体的向量表示
        
        Args:
            document_entities: 文档实体列表的列表，每个文档包含多个实体
            
        Returns:
            document_vectors: 文档实体向量列表
        """
        return [self.get_entity_vectors(entities) for entities in document_entities]

    def save_vectors(self, vectors: Dict[str, np.ndarray], output_path: str):
        """保存实体向量到文件
        
        Args:
            vectors: 实体向量字典
            output_path: 输出文件路径
        """
        np.save(output_path, vectors)
    
    @staticmethod
    def load_vectors(input_path: str) -> Dict[str, np.ndarray]:
        """从文件加载实体向量
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            vectors: 实体向量字典
        """
        return np.load(input_path, allow_pickle=True).item()
