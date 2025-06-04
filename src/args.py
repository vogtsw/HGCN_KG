import os
import models
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # 数据相关参数
    parser.add_argument('--data_dir', default='data/exaapd', type=str, help='数据目录路径')
    parser.add_argument('--dataset', default='exAAPD_hs', type=str, help='数据集名称')
    parser.add_argument('--max_seq_length', default=128, type=int, help='最大序列长度')
    
    # 模型相关参数
    parser.add_argument('--pretrained_model', default='SciBERT', type=str, help='预训练模型名称或路径')
    parser.add_argument('--hidden_dim', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout率')
    parser.add_argument('--num_labels', default=2, type=int, help='标签数量')
    
    # 训练相关参数
    parser.add_argument('--train_batch_size', default=32, type=int, help='训练批次大小')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='评估批次大小')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='学习率')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='训练轮数')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='预热步数比例')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度累积步数')
    
    # KG相关参数
    parser.add_argument('--wiki2vec_model_path', default='models/wiki2vec/enwiki_20180420_300d.pkl', type=str, help='Wikipedia2Vec模型路径')
    parser.add_argument('--vector_dim', default=300, type=int, help='实体向量维度')
    
    # 其他参数
    parser.add_argument('--do_train', action='store_true', help='是否训练')
    parser.add_argument('--do_eval', action='store_true', help='是否评估')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--no_cuda', action='store_true', help='是否不使用CUDA')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--save_path', default='checkpoints/kg_enhanced', type=str, help='模型保存路径')
    parser.add_argument('--trained_model', default=None, type=str, help='预训练模型路径')
    parser.add_argument('--fp16', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的本地rank')
    
    args = parser.parse_args()
    return args
