import os
import tqdm
import torch.nn as nn
import copy
import torch
import math
import torch.nn.functional as F
import numpy as np
from datasets.bert_processors.abstract_processor import BertProcessor, InputExample
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
     "Implements FFN equation."

     def __init__(self, d_model, d_ff, dropout=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.w_1 = nn.Linear(d_model, d_ff)
         self.w_2 = nn.Linear(d_ff, d_model)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value:torch.Size([30, 8, 10, 64])
    # decoder mask:torch.Size([30, 1, 9, 9])
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)  # torch.Size([30, 8, 64, 10])
    # torch.Size([30, 8, 10, 10])
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    if mask is not None:
        # decoder scores:torch.Size([30, 8, 9, 9]),
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 48=768//16
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value:torch.Size([2, 10, 768])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)    #2
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                          dropout=self.dropout)
         # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                  nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([2, 10, 768])
        return ret
#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.
class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x)))
        return ret

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      #多头注意力机制
        self.feed_forward = feed_forward    #前向神经网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        ret = self.sublayer[1](x, self.feed_forward)
        return ret

def top_k_graph(scores, g1,g2, h, k):
    num_nodes = g1.shape[1]
    values, idx = torch.topk(scores, max(2, k))
    new_h = []
    for i in range(4):
        new_h.append(h[i, idx[i, :], :].unsqueeze(0))
    new_h = torch.cat([new_h[0], new_h[1],new_h[2], new_h[3]], 0)
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    g_sentence = []
    g_section = []
    for i in range(4):
        g11 = g1[i,idx[i, :],:]
        g11 = g11[:, idx[i,:]]
        g_section.append(g11.unsqueeze(0))
        g22 = g2[i,idx[i,:],:]
        g22 = g22[:, idx[i,:]]
        g_sentence.append(g22.unsqueeze(0))
    return torch.cat([g_section[0],g_section[1],g_section[2],g_section[3]],0),torch.cat([g_sentence[0],g_sentence[1],g_sentence[2],g_sentence[3]],0), new_h

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class Pool(nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g1,g2, h, section_feature):
        Z = self.drop(h)
        weights = torch.max(torch.matmul(h, section_feature.transpose(1, 2)),dim=2)[0]
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g1,g2, h, self.k)

class Rea(nn.Module):
    def __init__(self):
        super(Rea, self).__init__()

        self.f1 = nn.Linear(768, 768)
        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)
        self.f4 = nn.Linear(768, 768)

        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x,section_mask_full,sentence_mask_full):
        att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28
        dia_att = F.softmax(att.masked_fill(section_mask_full == 0, -1e9), dim=2)
        prior_att = F.softmax((2*section_mask_full + 4*sentence_mask_full).masked_fill(section_mask_full == 0, -1e9), dim=2)
        fusion_att =  F.softmax((dia_att * prior_att).masked_fill(section_mask_full == 0, -1e9), dim=2)
        graph_r = self.dropout1(fusion_att)
        g_x1 = self.f3(torch.matmul(graph_r, x))
        no_self_att = F.softmax(att.masked_fill(section_mask_full == 1, -1e9), dim=2)
        graph_r1 = self.dropout1(no_self_att)
        g_x2 = self.f4(torch.matmul(graph_r1, g_x1))
        g_x = g_x1  + g_x2
        return g_x

class Rea_sec(nn.Module):
    def __init__(self):
        super(Rea_sec, self).__init__()

        self.f1 = nn.Linear(768, 768)
        self.f2 = nn.Linear(768, 768)
        self.f3 = nn.Linear(768, 768)
        self.dropout1 = nn.Dropout(0.1)

    def forward(self, x):
        att = torch.matmul(self.f1(x), self.f2(x).transpose(1, 2)) / 28
        att = F.softmax(att, dim=2)
        graph_r = self.dropout1(att)
        g_x = self.f3(torch.matmul(graph_r, x))
        return g_x

class DecoupledGraphPooling(customizedModule):
    def __init__(self,pooling_node,feature_dim,dropout = None):
        super(DecoupledGraphPooling, self).__init__()
        self.Rea = Rea()
        self.poolings = Pool(pooling_node, feature_dim, dropout)
        self.init_mBloSA()
        self.Rea_sec = Rea_sec()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, token_feature, section_feature, section_mask, sentence_mask):
        DGPN_output = self.Rea(token_feature,section_mask, sentence_mask)
        G = F.sigmoid(self.g_W1(token_feature) + self.g_W2(DGPN_output) + self.g_b)
        attention_output = G * token_feature + (1 - G) * DGPN_output
        new_sec_fea = self.Rea_sec(section_feature)
        section_mask, sentence_mask, new_att = self.poolings(section_mask, sentence_mask, attention_output, new_sec_fea)
        return section_mask,sentence_mask,new_att,new_sec_fea

class exAAPDProcessor_has_structure(BertProcessor):
    NAME = 'exAAPD'
    NUM_CLASSES = 54
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD', 'exAAPD_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD', 'exAAPD_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD', 'exAAPD_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = ['None']*7
            guid = "%s-%s" % (set_type, i)
            number = 0
            for num,(key,value) in enumerate(eval(line[1]).items()):
                if number < 7 and key != 'title':  #exAAPD
                    text[number] = value
                    number += 1
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

class exPFDProcessor_has_structure(BertProcessor):
    NAME = 'exPFD'
    NUM_CLASSES = 7
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD', 'exPFD_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD', 'exPFD_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD', 'exPFD_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = ['None']*4
            guid = "%s-%s" % (set_type, i)
            number = 0
            for num,(key,value) in enumerate(eval(line[1]).items()):
                if number < 4:                       #exPFD
                    text[number] = value
                    number += 1
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

class exLitCovidProcessor_has_structure(BertProcessor):
    NAME = 'exLitCovid'
    NUM_CLASSES = 8
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid', 'exLitCovid_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid', 'exLitCovid_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid', 'exLitCovid_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = ['None']*5
            guid = "%s-%s" % (set_type, i)
            number = 0
            for num,(key,value) in enumerate(eval(line[1]).items()):
                if number < 5:                       #exLitCovid
                    text[number] = value
                    number += 1
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

class exMeSHProcessor_has_structure(BertProcessor):
    NAME = 'exMeSH'
    NUM_CLASSES = 11
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH', 'Mesh_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH', 'Mesh_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH', 'Mesh_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text = ['None']*8
            guid = "%s-%s" % (set_type, i)
            number = 0
            for num,(key,value) in enumerate(eval(line[1]).items()):
                if number < 8:                       #exMeSH
                    text[number] = value
                    number += 1
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples

class exAAPDProcessor_no_structure(BertProcessor):
    NAME = 'exAAPD'
    NUM_CLASSES = 54
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD-full', 'exAAPD_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD-full', 'exAAPD_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exAAPD-full', 'exAAPD_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class exPFDProcessor_no_structure(BertProcessor):
    NAME = 'exPFD'
    NUM_CLASSES = 7
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD-full', 'exPFD_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD-full', 'exPFD_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exPFD-full', 'exPFD_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class exLitCovidProcessor_no_structure(BertProcessor):
    NAME = 'exLitCovid'
    NUM_CLASSES = 8
    IS_MULTILABEL = True

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid-full', 'exLitCovid_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid-full', 'exLitCovid_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exLitCovid-full', 'exLitCovid_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class exMeSHProcessor_no_structure(BertProcessor):
    NAME = 'exMeSH'
    NUM_CLASSES = 11
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH-full', 'exMeSH_train.json')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH-full', 'exMeSH_dev.json')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'exMeSH-full', 'exMeSH_test.json')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples