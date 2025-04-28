# import warnings
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn import metrics
# from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# from tqdm import tqdm
#
# from datasets.bert_processors.abstract_processor import convert_examples_to_features, \
#     convert_examples_to_hierarchical_features
# from utils.preprocessing import pad_input_matrix
#
# # Suppress warnings from sklearn.metrics
# warnings.filterwarnings('ignore')
#
#
# class BertEvaluator(object):
#     def __init__(self, model, processor, tokenizer, args, split='dev'):
#         self.args = args
#         self.model = model
#         self.processor = processor
#         self.tokenizer = tokenizer
#
#         if split == 'test':
#             self.eval_examples = self.processor.get_test_examples(args.data_dir)
#         else:
#             self.eval_examples = self.processor.get_dev_examples(args.data_dir)
#
#     def get_scores(self, silent=False):
#
#         eval_features = convert_examples_to_features(
#             self.eval_examples, self.args.max_seq_length, self.tokenizer)
#
#         unpadded_input_ids = [f.input_ids for f in eval_features]
#         unpadded_input_mask = [f.input_mask for f in eval_features]
#         unpadded_segment_ids = [f.segment_ids for f in eval_features]
#         unpadded_sentence_mask = [f.sentence_mask for f in eval_features]
#
#         padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
#         padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
#         padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
#         padded_sentence_mask = torch.tensor(unpadded_sentence_mask, dtype=torch.long)
#         label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
#
#         eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids,padded_sentence_mask, label_ids)
#         # eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids,
#         #                           label_ids)
#         eval_sampler = SequentialSampler(eval_data)
#         eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.train_batch_size)
#
#         self.model.eval()
#
#         total_loss = 0
#         nb_eval_steps, nb_eval_examples = 0, 0
#         predicted_labels, target_labels = list(), list()
#
#         for input_ids, input_mask, segment_ids, sentence_mask,label_ids in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
#             input_ids = input_ids.to(self.args.device)
#             input_mask = input_mask.to(self.args.device)
#             segment_ids = segment_ids.to(self.args.device)
#             sentence_mask = sentence_mask.to(self.args.device)
#             label_ids = label_ids.to(self.args.device)
#
#             with torch.no_grad():
#                 logits = self.model(input_ids.view(-1, 256), segment_ids.view(-1, 256), input_mask.view(-1, 256),sentence_mask)
#                 # logits = self.model(input_ids.view(-1, 256), segment_ids.view(-1, 256), input_mask.view(-1, 256))
#                 # logits = self.model(input_ids.view(-1, 256), input_mask.view(-1, 256), segment_ids.view(-1, 256))[0]
#
#             if self.args.is_multilabel:
#                 # logits = torch.sum(logits.view(-1, 10, 54), dim=1)
#                 predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
#                 target_labels.extend(label_ids.cpu().detach().numpy())
#                 loss = F.binary_cross_entropy_with_logits(logits, label_ids.float(), size_average=False)
#             else:
#                 predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
#                 target_labels.extend(torch.argmax(label_ids, dim=1).cpu().detach().numpy())
#                 loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))
#
#             if self.args.n_gpu > 1:
#                 loss = loss.mean()
#             if self.args.gradient_accumulation_steps > 1:
#                 loss = loss / self.args.gradient_accumulation_steps
#             total_loss += loss.item()
#
#             nb_eval_examples += input_ids.size(0)
#             nb_eval_steps += 1
#
#         predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
#         accuracy = metrics.accuracy_score(target_labels, predicted_labels)
#         precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
#         recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
#         f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
#         avg_loss = total_loss / nb_eval_steps
#
#         return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']


import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import os
from datasets.bert_processors.abstract_processor import convert_examples_to_features_hs,convert_examples_to_features_ns
from utils.preprocessing import pad_input_matrix

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class BertEvaluator(object):
    def __init__(self, model, processor, tokenizer, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        if split == 'test':
            self.eval_examples = self.processor.get_test_examples(args.data_dir)
            # self.eval_examples1 = np.load('/home/ltf/code/data/SDC/exPFD/exPFD_test_TKDE_4.npy',allow_pickle=True).tolist()
        else:
            self.eval_examples = self.processor.get_dev_examples(args.data_dir)
            # self.eval_examples1 = np.load('/home/ltf/code/data/SDC/exPFD/exPFD_dev_TKDE_4.npy',allow_pickle=True).tolist()

    def get_scores(self, silent=False):
        if self.args.is_hierarchical:
            eval_features = convert_examples_to_features_hs(
                self.eval_examples, self.args.max_seq_length, self.tokenizer)
            # eval_features = self.eval_examples1
        else:
            eval_features = convert_examples_to_features_ns(
                self.eval_examples, self.args.max_seq_length, self.tokenizer)

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]
        unpadded_sentence_mask = [f.sentence_mask for f in eval_features]

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        padded_sentence_mask = torch.tensor(unpadded_sentence_mask, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, padded_sentence_mask, label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.train_batch_size,drop_last = True)

        self.model.eval()

        total_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels, target_labels = list(), list()

        for input_ids, input_mask, segment_ids,sentence_mask,label_ids in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            sentence_mask = sentence_mask.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            with torch.no_grad():
                logits = self.model(input_ids.view(-1, self.args.max_seq_length), segment_ids.view(-1, self.args.max_seq_length), input_mask.view(-1, self.args.max_seq_length),sentence_mask)

            if self.args.is_multilabel:
                predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids.cpu().detach().numpy())
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float(), size_average=False)
            else:
                predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
                target_labels.extend(torch.argmax(label_ids, dim=1).cpu().detach().numpy())
                loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            total_loss += loss.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
