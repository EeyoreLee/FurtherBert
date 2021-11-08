# -*- encoding: utf-8 -*-
'''
@create_time: 2021/11/03 14:34:27
@author: lichunyu
'''

from collections import OrderedDict
import argparse

import torch
from transformers import BertConfig, BertModel
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
)
args = parser.parse_args()
HIDDEN_SIZE = args.hidden_size

config = BertConfig(
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4*HIDDEN_SIZE
)
model = BertModel(config)

output_dir = get_last_checkpoint('./model')
fp32_model = load_state_dict_from_zero_checkpoint(model, output_dir)
torch.save(fp32_model.state_dict(), './model/bert_large.pth')
pass