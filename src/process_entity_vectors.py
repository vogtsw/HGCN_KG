import os
import sys
import torch
import json
from tqdm import tqdm
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
from utils.entity_vector_processor import EntityVectorProcessor
from args import get_args

def is_valid_entity(text: str, type: str) -> bool:
    """检查实体是否有效
    
    Args:
        text: 实体文本
        type: 实体类型
        
    Returns:
        is_valid: 是否是有效实体
    """
    # 停用词列表
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "this", "that", "these", "those", "i", "you", "he", "she", "it", "they",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "would", "should", "could", "might", "must", "shall", "will"
    }
    
    # 常见动词列表
    common_verbs = {
        "go", "goes", "went", "gone", "make", "makes", "made",
        "take", "takes", "took", "taken", "give", "gives", "gave",
        "find", "finds", "found", "think", "thinks", "thought",
        "show", "shows", "shown", "need", "needs", "needed",
        "try", "tries", "tried", "call", "calls", "called"
    }
    
    # 检查是否包含无效字符
    if any(c in text for c in [".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]", "{", "}", "-", "_", "+", "=", "/", "\\", "|", "<", ">", "$", "#", "@", "%", "^", "&", "*", "~", "`"]):
        return False
    
    # 检查长度
    if len(text) < 2 or len(text) > 50:
        return False
    
    # 将文本分词
    words = text.lower().split()
    
    # 检查是否全是停用词
    if all(word in stopwords for word in words):
        return False
    
    # 专业术语词典
    tech_terms = {
        "quantum", "computing", "algorithm", "neural", "network", "deep", "learning",
        "machine", "artificial", "intelligence", "data", "analysis", "processing",
        "computational", "computer", "system", "software", "hardware", "model",
        "training", "classification", "recognition", "detection", "prediction",
        "optimization", "performance", "accuracy", "efficiency", "implementation",
        "framework", "architecture", "platform", "interface", "protocol",
        "medical", "clinical", "diagnostic", "therapeutic", "treatment",
        "patient", "disease", "symptom", "condition", "therapy", "drug",
        "medicine", "pharmaceutical", "biotechnology", "genomics", "proteomics",
        "imaging", "scan", "mri", "ct", "ultrasound", "xray", "radiation",
        "energy", "power", "electric", "electronic", "electrical", "circuit",
        "voltage", "current", "resistance", "semiconductor", "transistor",
        "renewable", "sustainable", "solar", "wind", "hydro", "thermal",
        "battery", "storage", "grid", "transmission", "distribution",
        "research", "science", "scientific", "technology", "technical",
        "experiment", "experimental", "study", "method", "methodology",
        "result", "approach", "solution", "problem", "challenge",
        "novel", "innovative", "advanced", "modern", "state-of-the-art",
        "hybrid", "integrated", "smart", "intelligent", "automated",
        "we", "our", "they", "their", "these", "those", "some", "many",
        "paper", "work", "project", "task", "goal", "objective",
        "propose", "present", "introduce", "develop", "implement",
        "improve", "enhance", "optimize", "evaluate", "validate",
        "demonstrate", "show", "prove", "verify", "test"
    }
    
    # 根据实体类型进行特定检查
    if type == "PER":
        # 人名通常是首字母大写的
        if not text[0].isupper():
            return False
        # 人名不应该是技术术语
        if any(word.lower() in tech_terms for word in words):
            return False
        # 人名不应该包含常见动词
        if any(word in common_verbs for word in words):
            return False
    
    elif type == "ORG":
        # 组织名通常是首字母大写的
        if not text[0].isupper():
            return False
        # 组织名不应该是单个动词
        if len(words) == 1 and words[0] in common_verbs:
            return False
    
    elif type == "LOC":
        # 地点名通常是首字母大写的
        if not text[0].isupper():
            return False
        # 地点名不应该是动词
        if len(words) == 1 and words[0] in common_verbs:
            return False
    
    elif type == "MISC":
        # 对于技术术语，检查是否在词典中
        if not any(word.lower() in tech_terms for word in words):
            # 如果不在词典中，检查是否是有效的组合词
            if not any(word.lower().endswith(("ing", "tion", "sion", "ment", "ity", "ness", "ance", "ence")) for word in words):
                return False
    
    return True

def post_process_entity(entity: Dict) -> Optional[Dict]:
    """对实体进行后处理
    
    Args:
        entity: 实体字典
        
    Returns:
        processed_entity: 处理后的实体字典，如果实体无效则返回None
    """
    # 特殊缩写和术语的映射
    term_mapping = {
        "mr scan": "MRI scan",
        "mri scan": "MRI scan",
        "ct scan": "CT scan",
        "ai": "AI",
        "ml": "ML",
        "dl": "DL",
        "nlp": "NLP",
        "cv": "CV",
        "quantum comp": "quantum computing",
        "deep learn": "deep learning",
        "machine learn": "machine learning",
        "neural net": "neural network",
        "artificial intel": "artificial intelligence"
    }
    
    # 统一大小写
    text = entity["text"]
    
    # 检查是否需要替换为标准术语
    lower_text = text.lower()
    for term, replacement in term_mapping.items():
        if lower_text == term or lower_text.startswith(term + " ") or lower_text.endswith(" " + term):
            text = replacement
            break
    
    # 如果是技术术语（MISC类型），统一使用标准形式
    if entity["type"] == "MISC":
        # 对于缩写词，保持大写
        if len(text) <= 4 and text.isupper():
            pass
        # 对于普通术语，只保持首字母大写
        elif text[0].isupper():
            text = text[0] + text[1:].lower()
    
    # 更新实体文本
    entity["text"] = text
    
    # 验证处理后的实体是否有效
    if is_valid_entity(text, entity["type"]):
        return entity
    return None

def extract_entities_from_text(text: str, 
                             model: BertForTokenClassification,
                             tokenizer: BertTokenizer,
                             args) -> List[Dict]:
    """从文本中提取实体
    
    Args:
        text: 输入文本
        model: BERT模型
        tokenizer: BERT分词器
        args: 参数
        
    Returns:
        entities: 实体列表
    """
    # 将文本转换为模型输入格式
    tokens = tokenizer.tokenize(text)
    if len(tokens) > args.max_seq_length - 2:
        tokens = tokens[:(args.max_seq_length - 2)]
    
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    
    # Padding
    padding = [0] * (args.max_seq_length - len(input_ids))
    input_ids += padding
    attention_mask += padding
    token_type_ids += padding
    
    # 转换为tensor
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(args.device)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(args.device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    # 解码预测结果
    label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    predicted_labels = [label_list[p] for p in predictions[:len(tokens)]]
    
    # 提取实体
    entities = []
    current_entity = None
    
    for token, label in zip(tokens[1:-1], predicted_labels[1:-1]):  # 跳过[CLS]和[SEP]
        if label.startswith("B-"):
            if current_entity:
                processed_entity = post_process_entity(current_entity)
                if processed_entity:
                    entities.append(processed_entity)
            current_entity = {"type": label[2:], "text": token.replace("##", "")}
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
            if not token.startswith("##"):
                current_entity["text"] += " "
            current_entity["text"] += token.replace("##", "")
        else:
            if current_entity:
                processed_entity = post_process_entity(current_entity)
                if processed_entity:
                    entities.append(processed_entity)
                current_entity = None
    
    if current_entity:
        processed_entity = post_process_entity(current_entity)
        if processed_entity:
            entities.append(processed_entity)
    
    return entities

def main():
    args = get_args()
    
    # 初始化BERT模型和分词器
    print("Initializing BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)
    model.to(device)
    
    # 加载文档
    print("Loading documents...")
    with open(os.path.join(args.data_dir, "text.txt"), "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f]
    
    # 提取实体
    print("Extracting entities...")
    all_entities = []
    for doc in tqdm(documents):
        entities = extract_entities_from_text(doc, model, tokenizer, args)
        all_entities.append(entities)
    
    # 保存提取的实体
    with open(os.path.join(args.data_dir, "extracted_entities.json"), "w", encoding="utf-8") as f:
        json.dump(all_entities, f, ensure_ascii=False, indent=2)
    
    # 初始化实体向量处理器
    print("Getting entity vectors...")
    entity_processor = EntityVectorProcessor(args.wiki2vec_model_path)
    
    # 获取实体向量
    document_vectors = entity_processor.get_document_entity_vectors(all_entities)
    
    # 保存实体向量
    entity_processor.save_vectors(
        {f"doc_{i}": vectors for i, vectors in enumerate(document_vectors)},
        os.path.join(args.data_dir, "entity_vectors.npy")
    )
    
    total_entities = sum(len(doc_entities) for doc_entities in all_entities)
    print(f"Processed {len(documents)} documents")
    print(f"Extracted {total_entities} entities")
    print(f"Entity vectors saved to {os.path.join(args.data_dir, 'entity_vectors.npy')}")

if __name__ == "__main__":
    main()
