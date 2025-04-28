# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import scipy.linalg

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
import torch.nn as nn
from tqdm import tqdm, trange
from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_trainer import BertTrainer
from transformers import WarmupLinearSchedule
from transformers.modeling_bert import BertForSequenceClassification, BertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW
from args import get_args
from pytorch_pretrained_bert import BertModel
from datasets.bert_processors.aapd_processor import exAAPDProcessor_has_structure,exPFDProcessor_has_structure,exLitCovidProcessor_has_structure,exMeSHProcessor_has_structure,
                                                    exAAPDProcessor_no_structure,exPFDProcessor_no_structure,exLitCovidProcessor_no_structure,exMeSHProcessor_no_structure
from common.constants import *
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_split(model, processor, tokenizer, args, split='dev'):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split)
    accuracy, precision, recall, f1, avg_loss = evaluator.get_scores(silent=True)[0]
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl

class ClassifyModel(customizedModule):
    def __init__(self, pretrained_model_name_or_path, args, device, is_lock=False):
        super(ClassifyModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.args = args
        self.DGP = nn.ModuleList()
        self.DGP.append(DecoupledGraphPooling(508, 768, 0.1))
        self.DGP.append(DecoupledGraphPooling(254, 768, 0.1))
        self.DGP.append(DecoupledGraphPooling(127, 768, 0.1))
        self.classifier = nn.Linear(768, self.args.num_labels)
        self.zero = torch.zeros((254, 254)).to(device)
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sentence_mask=None, label=None, ):
        all_token_feature, pooled_feature = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)

        token_feature = all_token_feature[:, 1:-1, :]  # 获取段中所有单词的特征
        token_feature = torch.reshape(token_feature,(-1, 8 * (self.args.max_seq_length - 2), 768))  # batch * word num * feature dim
        # ******************************************************************************************************************************************************
        # 生成mask
        h_mask = sentence_mask.unsqueeze(-1).expand(self.args.train_batch_size, 8, self.args.max_seq_length - 2,self.args.max_seq_length - 2)
        v_mask = sentence_mask.unsqueeze(-2).expand(self.args.train_batch_size, 8, self.args.max_seq_length - 2,self.args.max_seq_length - 2)
        sentence_mask_full = torch.stack([torch.cat((torch.cat((i[0, :], self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, i[1, :], self.zero, self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, i[2, :], self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, i[3, :], self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, i[4, :], self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, i[5, :], self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, i[6, :], self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, i[7, :]), 0)), 1) for i in (h_mask == v_mask)])
        attention_mask_1 = torch.tensor(attention_mask.view(-1, 8, self.args.max_seq_length).unsqueeze(-1),dtype=torch.float)
        attention_mask_2 = torch.matmul(attention_mask_1, attention_mask_1.transpose(2, 3))[:, :, 1:-1, 1:-1]
        section_mask_full = torch.stack([torch.cat((torch.cat((i[0, :], self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, i[1, :], self.zero, self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, i[2, :], self.zero, self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, i[3, :], self.zero, self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, i[4, :], self.zero, self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, i[5, :], self.zero, self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, i[6, :], self.zero), 0),
                                                     torch.cat((self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, self.zero, i[7, :]), 0)), 1) for i in attention_mask_2])
        # *************************************************************************************************************************************************
        # 解耦图池化模块*******
        section_mask, sentence_mask, new_att1, new_sec_fea1 = self.DGP[0](token_feature,pooled_feature.view(-1, 8, 768),section_mask_full, sentence_mask_full)
        section_mask, sentence_mask, new_att2, new_sec_fea2 = self.DGP[1](new_att1, new_sec_fea1, section_mask,sentence_mask)
        section_mask, sentence_mask, new_att3, new_sec_fea3 = self.DGP[2](new_att2, new_sec_fea2, section_mask,sentence_mask)

        u = torch.max(torch.cat([torch.max(new_att1, dim=1)[0].unsqueeze(1),
                                 torch.max(new_att2, dim=1)[0].unsqueeze(1),
                                 torch.max(new_att3, dim=1)[0].unsqueeze(1),
                                 torch.max(new_sec_fea3, dim=1)[0].unsqueeze(1)], dim=1), dim=1)[0]

        logits = self.classifier(u)
        return logits

def main():
    #Set default configuration in args.py
    args = get_args()
    dataset_map = {'exAAPD_hs': exAAPDProcessor_has_structure,
                   'exAAPD_ns': exAAPDProcessor_no_structure
                   'exPFD_hs': exPFDProcessor_has_structure,
                   'exPFD_ns': exPFDProcessor_no_structure
                   'exLitCovid_hs': exLitCovidProcessor_has_structure,
                   'exLitCovid_ns': exLitCovidProcessor_no_structure
                   'exMeSH_hs': exMeSHProcessor_has_structure,
                   'exMeSH_ns': exMeSHProcessor_no_structure}

    output_modes = {"rte": "classification"}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    args.device = device
    args.n_gpu = n_gpu  # 1
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES  # 7
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL  # False
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    args.is_hierarchical = True
    processor = dataset_map[args.dataset]()

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    if args.pretrained_model == 'SciBERT':
        pretrain_model_dir ='/home/ltf/code/data/scibert_scivocab_uncased/'
        model = ClassifyModel(pretrain_model_dir, args=args, device=device, is_lock=False)
        tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if not args.trained_model:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install NVIDIA Apex for FP16 training")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
            #Transformer version == 2.1.1
            scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                             warmup_steps=args.warmup_proportion * num_train_optimization_steps)
            # Transformer version == 3.1.0/4.6.0
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
            #                                             num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)
        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = torch.load('/home/ltf/code/data/best_weight/exMeSH/2021-05-09_16-20-31.pt')   #best weight for exMeSH
        model = model.to(device)

    evaluate_split(model, processor, tokenizer, args, split='dev')
    evaluate_split(model, processor, tokenizer, args, split='test')


if __name__ == "__main__":
    main()
