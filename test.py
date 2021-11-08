# -*- encoding: utf-8 -*-
'''
@create_time: 2021/11/03 15:19:30
@author: lichunyu
'''

import torch
from transformers import BertConfig, BertModel


HIDDEN_SIZE = 2256

config = BertConfig(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4*HIDDEN_SIZE
)
model = BertModel(config)
state_dict = torch.load('./model/bert_large.pth')
model.load_state_dict(state_dict)
pass