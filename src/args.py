import os
import models
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-chinese', type=str)
    parser.add_argument('--model_dir', default='models/ner', type=str)
    parser.add_argument('--data_dir', default='data/ner', type=str)
    parser.add_argument('--task', default='ner', type=str)
    parser.add_argument('--max_seq_length', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_labels', type=int, default=9)  # NER标签数量
    parser.add_argument('--wiki2vec_model_path',
                      default='models/wiki2vec/enwiki_20180420_300d.pkl',
                      type=str,
                      help='Path to pre-trained Wikipedia2Vec model')
    parser.add_argument('--vector_dim',
                      default=300,
                      type=int,
                      help='Dimension of entity vectors')
    args = parser.parse_args()
    return args
