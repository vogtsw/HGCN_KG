import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import torch
from pytorch_pretrained_bert import BertTokenizer
from models.ner_model import NERModel
from datasets.bert_processors.ner_processor import NERProcessor
from args import get_args

def predict_ner(text, model, processor, tokenizer, args):
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
        _, logits = model(input_ids, token_type_ids, attention_mask)
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    
    # 解码预测结果
    label_map = {i: label for label, i in processor.label_map.items()}
    predicted_labels = [label_map[p] for p in predictions[:len(tokens)]]
    
    # 将预测结果与原文本对齐
    results = []
    current_entity = None
    
    for token, label in zip(tokens[1:-1], predicted_labels[1:-1]):  # 跳过[CLS]和[SEP]
        if label.startswith("B-"):
            if current_entity:
                results.append(current_entity)
            current_entity = {"type": label[2:], "text": token, "start": len(results)}
        elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
            current_entity["text"] += token
        else:
            if current_entity:
                results.append(current_entity)
                current_entity = None
    
    if current_entity:
        results.append(current_entity)
    
    return results

def main():
    args = get_args()
    
    # 初始化processor、tokenizer和model
    processor = NERProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    model = NERModel(args.model, len(processor.get_labels()), device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'ner_model_final.bin')))
    model.to(device)
    
    # 测试文本
    test_texts = [
        "中国科学院的张伟在北京工作。",
        "他来自上海交通大学。",
        "李明在清华大学读书。",
        "王小明在腾讯公司工作。"
    ]
    
    for text in test_texts:
        print(f"\n输入文本: {text}")
        entities = predict_ner(text, model, processor, tokenizer, args)
        print("识别的实体:")
        for entity in entities:
            print(f"类型: {entity['type']}, 文本: {entity['text']}")

if __name__ == "__main__":
    main()
