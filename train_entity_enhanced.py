import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel

from common.evaluators.classification_evaluator import ClassificationEvaluator
from datasets.bert_processors.entity_enhanced_processor import EntityEnhancedProcessor
from models.entity_enhanced_hgcn import EntityEnhancedHGCN
from utils.seed import set_seed
from utils.args import get_args

def evaluate_split(model, data_loader, split):
    """评估模型在指定数据集上的性能"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 将数据移动到GPU（如果可用）
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            
            input_ids, attention_mask, entity_vectors, batch_labels = batch
            
            logits = model(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         entity_vectors=entity_vectors)
            
            loss = nn.CrossEntropyLoss()(logits, batch_labels)
            total_loss += loss.item()
            
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    return total_loss / len(data_loader), accuracy

def load_data(args):
    """加载数据"""
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(args.model)
    
    # 加载数据处理器
    processor = EntityEnhancedProcessor(
        entity_vector_path=os.path.join(args.data_dir, "entity_vectors.npy")
    )
    
    # 加载数据
    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    
    # 创建数据加载器
    train_features = processor.convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
    train_loader = processor.get_data_loader(train_features, args.batch_size, shuffle=True)
    
    dev_features = processor.convert_examples_to_features(dev_examples, args.max_seq_length, tokenizer)
    dev_loader = processor.get_data_loader(dev_features, args.batch_size)
    
    test_features = processor.convert_examples_to_features(test_examples, args.max_seq_length, tokenizer)
    test_loader = processor.get_data_loader(test_features, args.batch_size)
    
    return train_loader, dev_loader, test_loader

def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载数据
    train_loader, dev_loader, test_loader = load_data(args)
    
    # 创建模型
    config = {
        'encoder': None,  # 需要根据实际情况设置
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'target_size': args.num_labels,
        'model_name': args.model
    }
    
    # 初始化BERT编码器
    config['encoder'] = BertModel.from_pretrained(args.model)
    
    # 创建模型
    model = EntityEnhancedHGCN(config)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # 将数据移动到GPU（如果可用）
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            
            input_ids, attention_mask, entity_vectors, labels = batch
            
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, 
                         attention_mask=attention_mask, 
                         entity_vectors=entity_vectors)
            
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 评估
        train_loss, train_acc = evaluate_split(model, train_loader, "train")
        dev_loss, dev_acc = evaluate_split(model, dev_loader, "dev")
        test_loss, test_acc = evaluate_split(model, test_loader, "test")
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # 保存最佳模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pt"))

def main():
    args = get_args()
    train(args)

if __name__ == "__main__":
    main()
