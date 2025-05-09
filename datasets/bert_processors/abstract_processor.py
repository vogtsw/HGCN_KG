import csv

import sys
import numpy as np
from nltk.tokenize import sent_tokenize
import json
import re
import tqdm
import logging
import scipy.linalg
import torch
import pickle
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, entity_vectors=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            entity_vectors: (Optional) numpy array. The entity vectors for this example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.entity_vectors = entity_vectors


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, sentence_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sentence_mask = sentence_mask
        self.label_id = label_id

class BertProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the train set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the dev set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """
        Gets a collection of `InputExample`s for the test set
        :param data_dir:
        :return:
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Gets a list of possible labels in the dataset
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def _clean_str(cls,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
        Reads a Tab Separated Values (TSV) file
        :param input_file:
        :param quotechar:
        :return:
        """
        lines = []
        for line in open(input_file, 'r', encoding='utf-8'):
            text = line.split('\t')
            lines.append(text)
        return lines

def convert_examples_to_features_hs(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_sentence_mask = []
        for section in example.text_a:
            if section != 'None':
                tokens_sentence_mask = np.zeros(max_seq_length - 2)
                tokens_sentence = [line for line in sent_tokenize(section)]  # 可能有10个句子
                sentence_idx = 0
                n = 1
                for sentence in tokens_sentence:
                    sentence_length = len(tokenizer.tokenize(sentence))
                    if sentence_idx + sentence_length >= max_seq_length - 2:
                        sentence_length = max_seq_length - 2 - sentence_idx
                    tokens_sentence_mask[sentence_idx:sentence_idx + sentence_length] = n
                    sentence_idx += sentence_length
                    n += 1
                #***********************************************************************************************************
                token = tokenizer.tokenize(section)
                if len(token) > max_seq_length - 2:
                    token = token[:(max_seq_length - 2)]

                tokens = ["[CLS]"] + token + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                label_id = [float(x) for x in example.label]
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                all_sentence_mask.append(tokens_sentence_mask)
            else:
                all_input_ids.append([0] * max_seq_length)
                all_input_mask.append([0] * max_seq_length)
                all_segment_ids.append([0] * max_seq_length)
                all_sentence_mask.append(np.zeros(max_seq_length - 2))

        features.append(InputFeatures(input_ids=all_input_ids,
                                      input_mask=all_input_mask,
                                      segment_ids=all_segment_ids,
                                      sentence_mask = all_sentence_mask,
                                      label_id=label_id))

    return features

def convert_examples_to_features_ns(examples, max_seq_length, tokenizer, print_examples=False):
    """
    Loads a data file into a list of InputBatch objects
    :param examples:
    :param max_seq_length:
    :param tokenizer:
    :param print_examples:
    :return: a list of InputBatch objects
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_sentence_mask = []
        all_section = ['None'] * 100
        num_num = 0
        num = 0
        all_sentence = ''
        for sentence in sent_tokenize(example.text_a):
            if num + len(tokenizer.tokenize(sentence)) > max_seq_length - 2:
                all_section[num_num] = all_sentence
                num_num += 1
                if num_num >= 8:    # exAAPD-7,exPFD-4,exLitCovid-5,exMeSH-8
                    break
                all_sentence = sentence.strip()
                all_sentence += ' '
                num = len(tokenizer.tokenize(sentence))
                continue
            else:
                all_sentence += sentence.strip()
                all_sentence += ' '
            num += len(tokenizer.tokenize(sentence))
        for id, section in enumerate(all_section):
            if section != 'None':
                # 首先获得句子的mask-sentence mask
                # 首先初始化一个256*1的零矩阵
                tokens_sentence_mask = np.zeros(max_seq_length - 2)
                tokens_sentence = [line for line in sent_tokenize(section)]  # 可能有10个句子
                sentence_idx = 0
                n = 1
                for sentence in tokens_sentence:
                    sentence_length = len(tokenizer.tokenize(clean_str(sentence)))
                    if sentence_idx + sentence_length >= max_seq_length - 2:
                        sentence_length = max_seq_length - 2 - sentence_idx
                    tokens_sentence_mask[sentence_idx:sentence_idx + sentence_length] = n
                    sentence_idx += sentence_length
                    n += 1
                # ***********************************************************************************************************
                token = tokenizer.tokenize(section)
                if len(token) > max_seq_length - 2:
                    token = token[:(max_seq_length - 2)]

                tokens = ["[CLS]"] + token + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                label_id = [float(x) for x in example.label]
                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                all_sentence_mask.append(tokens_sentence_mask)
            else:
                all_input_ids.append([0] * max_seq_length)
                all_input_mask.append([0] * max_seq_length)
                all_segment_ids.append([0] * max_seq_length)
                all_sentence_mask.append(np.zeros(max_seq_length - 2))

        features.append(InputFeatures(input_ids=all_input_ids,
                                      input_mask=all_input_mask,
                                      segment_ids=all_segment_ids,
                                      sentence_mask=all_sentence_mask,
                                      label_id=label_id))

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length
    :param tokens_a:
    :param tokens_b:
    :param max_length:
    :return:
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
