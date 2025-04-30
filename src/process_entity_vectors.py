import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from transformers import BertTokenizer, BertForTokenClassification
from wikipedia2vec import Wikipedia2Vec

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

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

def combine_subword_tokens(tokens: List[str], predictions: np.ndarray, probabilities: np.ndarray, threshold: float = 0.3) -> List[Dict]:
    """合并子词标记并提取实体
    
    Args:
        tokens: 分词后的标记列表
        predictions: 每个标记的预测标签
        probabilities: 每个标记的预测概率
        threshold: 预测概率阈值
        
    Returns:
        entities: 实体列表
    """
    entities = []
    current_entity = None
    current_text = []
    prev_was_entity = False
    
    # 跳过[CLS]和[SEP]标记
    for i, (token, pred, prob) in enumerate(zip(tokens[1:-1], predictions[1:-1], probabilities[1:-1])):
        # 如果是子词标记（以##开头）
        is_subword = token.startswith("##")
        clean_token = token[2:] if is_subword else token
        
        # 只有当预测概率大于阈值时才考虑这个标记
        if prob > threshold:
            if pred == 1:  # B-tag
                # 保存之前的实体
                if current_entity and len(current_text) > 0:
                    current_entity["text"] = " ".join("".join(current_text).split())
                    if len(current_entity["text"]) > 1 and not current_entity["text"].lower() in [".", ",", "'", ":", ";", "-", "and", "or", "the", "a", "an"]:
                        entities.append(current_entity)
                # 开始新实体
                current_text = [clean_token]
                current_entity = {"text": "", "type": "ENTITY", "start": i, "end": i}
                prev_was_entity = True
            elif pred == 2:  # I-tag
                if prev_was_entity:
                    # 继续当前实体
                    if is_subword:
                        current_text.append(clean_token)
                    else:
                        current_text.append(" " + clean_token)
                    if current_entity:
                        current_entity["end"] = i
                else:
                    # 如果前一个不是实体，将其视为B-tag开始新实体
                    if current_entity and len(current_text) > 0:
                        current_entity["text"] = " ".join("".join(current_text).split())
                        if len(current_entity["text"]) > 1 and not current_entity["text"].lower() in [".", ",", "'", ":", ";", "-", "and", "or", "the", "a", "an"]:
                            entities.append(current_entity)
                    current_text = [clean_token]
                    current_entity = {"text": "", "type": "ENTITY", "start": i, "end": i}
                    prev_was_entity = True
            else:  # O-tag
                if current_entity and len(current_text) > 0:
                    current_entity["text"] = " ".join("".join(current_text).split())
                    if len(current_entity["text"]) > 1 and not current_entity["text"].lower() in [".", ",", "'", ":", ";", "-", "and", "or", "the", "a", "an"]:
                        entities.append(current_entity)
                current_entity = None
                current_text = []
                prev_was_entity = False
        else:
            if current_entity and len(current_text) > 0:
                current_entity["text"] = " ".join("".join(current_text).split())
                if len(current_entity["text"]) > 1 and not current_entity["text"].lower() in [".", ",", "'", ":", ";", "-", "and", "or", "the", "a", "an"]:
                    entities.append(current_entity)
            current_entity = None
            current_text = []
            prev_was_entity = False
    
    # 处理最后一个实体
    if current_entity and len(current_text) > 0:
        current_entity["text"] = " ".join("".join(current_text).split())
        if len(current_entity["text"]) > 1 and not current_entity["text"].lower() in [".", ",", "'", ":", ";", "-", "and", "or", "the", "a", "an"]:
            entities.append(current_entity)
    
    # 清理实体文本
    cleaned_entities = []
    for entity in entities:
        # 清理文本
        text = entity["text"].strip()
        # 移除特殊字符
        text = text.replace("#", "").strip()
        if text and len(text) > 1:
            entity["text"] = text
            cleaned_entities.append(entity)
    
    return cleaned_entities

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
    try:
        # 分词
        tokens = tokenizer.tokenize(text)
        
        # 如果文本太长，进行截断
        if len(tokens) > args.max_seq_length - 2:  # 考虑[CLS]和[SEP]
            tokens = tokens[:(args.max_seq_length - 2)]
        
        # 添加特殊标记
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding = [0] * (args.max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(args.device)
        
        # 获取模型输出
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, sequence_length, num_labels]
        
        # 获取预测结果
        probabilities = torch.softmax(logits, dim=-1)  # [batch_size, sequence_length, num_labels]
        predictions = torch.argmax(probabilities, dim=-1)  # [batch_size, sequence_length]
        predictions = predictions[0].detach().cpu().numpy()  # [sequence_length]
        max_probs = torch.max(probabilities[0], dim=-1)[0].detach().cpu().numpy()  # [sequence_length]
        
        # 打印调试信息
        print(f"\nProcessing text: {text[:100]}...")  # 只显示前100个字符
        print(f"Number of tokens: {len(tokens)}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Predictions shape: {predictions.shape}")
        
        # 确保predictions的长度与tokens的长度匹配
        token_predictions = predictions[:len(tokens)]
        token_probabilities = max_probs[:len(tokens)]
        
        # 如果predictions是标量，将其转换为长度为1的数组
        if np.isscalar(token_predictions):
            token_predictions = np.array([0] * len(tokens))  # 默认所有token都是O标签
            token_probabilities = np.array([1.0] * len(tokens))
        
        # 打印每个token和对应的预测标签
        print("\nToken predictions:")
        for token, pred, prob in list(zip(tokens, token_predictions, token_probabilities))[:10]:  # 只显示前10个token的预测
            print(f"{token}: {pred} (prob={prob:.4f})")
        
        # 合并子词标记并提取实体
        entities = combine_subword_tokens(tokens, token_predictions, token_probabilities)
        
        # 打印识别出的实体
        print(f"Found {len(entities)} entities: {entities[:3]}...")  # 只显示前3个实体
        
        return entities
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        print(f"Text length: {len(text)}")
        print(f"Text preview: {text[:100]}...")
        return []

def load_json_data(file_path, max_samples=None):
    """加载JSON数据，处理可能的二进制前缀
    
    Args:
        file_path: JSON文件路径
        max_samples: 最大样本数，如果为None则加载所有样本
        
    Returns:
        data: JSON数据列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        start_idx = content.find('{')
        content = content[start_idx:]
        
        # 尝试加载JSON数据
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            # 如果是多个JSON对象，尝试逐行解析
            lines = content.strip().split('\n')
            data = []
            for line in lines:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    
    # 如果指定了最大样本数，只返回前max_samples个样本
    if max_samples is not None:
        data = data[:max_samples]
    
    return data

def main():
    # 解析参数
    args = get_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    # 加载BERT模型
    print("Loading BERT model...")
    model_name = 'bert-base-cased'  # 使用预训练的cased模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=args.num_labels)
    model.to(device)
    
    # 加载Wikipedia2Vec模型
    print("Loading Wikipedia2Vec model...")
    wiki2vec = Wikipedia2Vec.load(args.wiki2vec_model_path)
    
    # 加载训练数据
    train_file = os.path.join(args.data_dir, "exAAPD_train.json")
    train_data = load_json_data(train_file)
    
    # 创建输出目录
    output_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文档
    all_entities = []
    all_vectors = []
    
    for doc in tqdm(train_data, desc="Processing documents"):
        # 获取文本
        text = doc.get("text", "") or doc.get("title", "")
        if not isinstance(text, str) or not text.strip():
            continue
            
        # 提取实体
        entities = extract_entities_from_text(text, model, tokenizer, args)
        if not entities:
            continue
            
        # 获取实体向量
        doc_vectors = []
        for entity in entities:
            try:
                vector = wiki2vec.get_entity_vector(entity["text"])
                if vector is not None:
                    doc_vectors.append(vector)
            except (KeyError, ValueError) as e:
                continue
        
        if doc_vectors:
            all_entities.append(entities)
            all_vectors.append(doc_vectors)
    
    # 保存结果
    print("Saving vectors...")
    with open(os.path.join(output_dir, "train_entities.json"), "w", encoding="utf-8") as f:
        json.dump(all_entities, f, indent=2, ensure_ascii=False)
    
    np.save(os.path.join(output_dir, "train_entity_vectors.npy"), np.array(all_vectors, dtype=object))
    
    print(f"Done! Processed {len(all_entities)} documents, extracted {sum(len(doc_entities) for doc_entities in all_entities)} entities")
    
if __name__ == "__main__":
    main()
