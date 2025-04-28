import os
import sys
import random
import logging
import torch
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from pytorch_pretrained_bert import BertTokenizer
from torch.optim import Adam
from common.evaluators.ner_evaluator import NEREvaluator
from common.trainers.ner_trainer import NERTrainer
from datasets.bert_processors.ner_processor import NERProcessor
from models.ner_model import NERModel
from args import get_args

def train(args):
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize processor, tokenizer and model
    processor = NERProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    model = NERModel(args.model, len(processor.get_labels()), device)
    model.to(device)

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr)
    scheduler = None  

    trainer = NERTrainer(model, optimizer, processor, scheduler, tokenizer, args)
    trainer.train()
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'ner_model_final.bin'))

def evaluate(args):
    processor = NERProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = NERModel(args.model, len(processor.get_labels()), args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'ner_model_final.bin')))
    model.to(args.device)
    
    evaluator = NEREvaluator(model, processor, tokenizer, args)
    result = evaluator.get_scores()
    print(result)

def main():
    args = get_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    if args.do_train:
        train(args)
        
    if args.do_eval:
        evaluate(args)

if __name__ == '__main__':
    main()
