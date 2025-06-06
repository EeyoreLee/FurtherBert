import os
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertModel, BertConfig, Trainer, AdamW, get_scheduler, TrainingArguments
from transformers.trainer_utils import is_main_process
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import argparse
from tqdm import tqdm
import numpy as np


class DataNumpyDataset(Dataset):
    def __init__(self, ars=2):
        self.dataset_size = ars
        self.npy = open("./code/data.npy", "rb")
        data = np.load(self.npy, allow_pickle=True)
        pt_data = dict()
        pt_data["input_ids"] = torch.LongTensor(data[0]["input_ids"]).cuda()
        pt_data["token_type_ids"] = torch.LongTensor(data[0]["token_type_ids"]).cuda()
        pt_data["attention_mask"] = torch.LongTensor(data[0]["attention_mask"]).cuda()
        pt_data['label'] = torch.LongTensor(data[1]).cuda()
        pt_data['masked_lm_ids'] = torch.LongTensor(data[2]).cuda()
        pt_data['masked_lm_positions'] = torch.LongTensor(data[3]).cuda()
        pt_data['masked_lm_weights'] = torch.LongTensor(data[4]).cuda()
        self.pt_data = pt_data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            self.npy.seek(0)
            raise IndexError
        data = {
            'input_ids': self.pt_data['input_ids'][idx],
            'token_type_ids': self.pt_data['token_type_ids'][idx],
            'attention_mask': self.pt_data['attention_mask'][idx],
            'label': self.pt_data['label'][idx],
            'masked_lm_ids': self.pt_data['masked_lm_ids'][idx],
            'masked_lm_positions': self.pt_data['masked_lm_positions'][idx],
            'masked_lm_weights': self.pt_data['masked_lm_weights'][idx],
        }
        return data

class BertPreTrainingHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.predictions = BertLMPredictionHead(
            hidden_size, vocab_size, hidden_act)
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_scores = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, hidden_act=nn.GELU()):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.output_bias

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sequence_output = self.decoder(sequence_output)
        return sequence_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act=nn.GELU()):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, sequence_output):
        sequence_output = self.dense(sequence_output)
        sequence_output = self.transform_act_fn(sequence_output)
        sequence_output = self.LayerNorm(sequence_output)
        return sequence_output


class BertFurtherModel(nn.Module):

    def __init__(self, bert_config, max_predictions_per_seq):
        super().__init__()
        self.config = bert_config
        self.bert = BertModel(bert_config)
        self.bert_pretrain_heads = BertPreTrainingHeads(bert_config.hidden_size, bert_config.vocab_size)
        self.max_predictions_per_seq = max_predictions_per_seq
        self.mlm_criterion = nn.CrossEntropyLoss(reduction="none")


        def get_masked_lm_loss(
                logit_blob,
                masked_lm_positions,
                masked_lm_labels,
                label_weights,
                max_predictions_per_seq,
        ):
            # gather valid position indices
            logit_blob = torch.gather(
                logit_blob,
                index=masked_lm_positions.unsqueeze(2).to(
                    dtype=torch.int64).repeat(1, 1, 30522),
                dim=1,
            )
            logit_blob = torch.reshape(logit_blob, [-1, 30522])
            label_id_blob = torch.reshape(masked_lm_labels, [-1])

            # The `positions` tensor might be zero-padded (if the sequence is too
            # short to have the maximum number of predictions). The `label_weights`
            # tensor has a value of 1.0 for every real prediction and 0.0 for the
            # padding predictions.
            pre_example_loss = self.mlm_criterion(logit_blob, label_id_blob.long())
            pre_example_loss = torch.reshape(
                pre_example_loss, [-1, max_predictions_per_seq])
            sum_label_weight = torch.sum(label_weights, dim=-1)
            sum_label_weight = sum_label_weight // label_weights.shape[0]
            numerator = torch.sum(pre_example_loss * label_weights)
            denominator = torch.sum(label_weights) + 1e-5
            loss = numerator / denominator
            return loss
        self.ns_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.masked_lm_criterion = get_masked_lm_loss

    def forward(self, input_ids, token_type_ids, attention_mask, label, masked_lm_ids, masked_lm_positions, masked_lm_weights, return_outputs=False, *args, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_scores, seq_relationship_scores = self.bert_pretrain_heads(outputs.last_hidden_state, outputs.pooler_output)
        next_sentence_loss = self.ns_criterion(seq_relationship_scores.view(-1, 2), label.long().view(-1))
        masked_lm_loss = self.masked_lm_criterion(prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights, max_predictions_per_seq=self.max_predictions_per_seq)
        total_loss = next_sentence_loss + masked_lm_loss
        return (total_loss, outputs) if return_outputs else (total_loss,)


class PreTrainer(Trainer):

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default=None)

    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="wiki_ofrecord_seq_len_128_example",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=2, help="The number of samples in an epoch cycle",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    args = parser.parse_args()

    _training_args = TrainingArguments(
        deepspeed='./code/ds_auto_config.json',
        output_dir='./model',
        do_train=True,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=10,
        remove_unused_columns=False,
        disable_tqdm=False,
        save_strategy='steps',
        save_total_limit=1,
        save_steps=20,
        do_eval=False
    )

    if is_main_process(_training_args.local_rank):
        print("Creating Dataloader")

    configuration = BertConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               num_attention_heads=args.num_attention_heads, intermediate_size=4 * args.hidden_size)

    model = BertFurtherModel(configuration, args.max_predictions_per_seq)
    # model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.adam_weight_decay)

    lr_scheduler = get_scheduler(
        "polynomial",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=300
    )

    data_dataset = DataNumpyDataset()

    trainer = PreTrainer(
        model=model,
        optimizers=(optimizer, lr_scheduler),
        args=_training_args,
        train_dataset=data_dataset
        )

    try:
        trainer.train()
    except Exception as e:
        print(e)
        sys.exit(1)

    if is_main_process(_training_args.local_rank):
        print('End of training .')
        print('\n')
        print('Start to save the model ...')
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
        torch.save(fp32_model.state_dict(), './model/bert_large_{}.pth'.format(HIDDEN_SIZE))

        print('End of saving .')
        print('')
        #print('final hidden size is {}'.format(HIDDEN_SIZE))
        sys.exit()
