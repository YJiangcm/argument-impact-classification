# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:21 2021

@author: JIANG Yuxin
"""


import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def process_text(x):
  if x[:2] == "['":
    x = x[2:]
  if x[-2:] == "']":
    x = x[:-2]
  return x
  
  
def map_impact_label(x):
    if x == "NOT_IMPACTFUL":
        return 0
    elif x == "MEDIUM_IMPACT":
        return 1
    else:
        return 2


def str2list(x):
    return ast.literal_eval(x)


def extract_context(n_context, context, stance_label):
    context = context[-n_context : ]
    stance_label = stance_label[1:][-n_context : ]
    contexts = ""
    for i in range(1, len(context)+1):
        contexts += stance_label[-i] + " " + context[-i] 
        if i != len(context):
            contexts += " "
    return contexts
    

def csv2df(data_path, n_context=1):
    data = pd.read_csv(data_path, encoding='utf-8').iloc[:, 1:5]
    data['text'] = data['text'].apply(process_text)
    data['stance_label'] = data['stance_label'].apply(str2list)
    data['context'] = data['context'].apply(str2list)
    data['context'] = data.apply(lambda x : extract_context(n_context, x['context'], x['stance_label']), axis=1)
    data['impact_label'] = data['impact_label'].apply(map_impact_label)
    return data


class AIC_Dataset(Dataset):
    def __init__(self, tokenizer, data_path, max_seq_len=128, n_context=1, is_labeling=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.n_context = n_context
        self.is_labeling = is_labeling
        self.input_ids, self.input_mask, self.segment_ids, self.impact_label_id = self.get_input(data_path)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.is_labeling:
            return self.input_ids[idx], self.input_mask[idx], self.segment_ids[idx], self.impact_label_id[idx]
        else:
            return self.input_ids[idx], self.input_mask[idx], self.segment_ids[idx]
        
    def get_input(self, data_path):
        df = csv2df(data_path, self.n_context)
        text = df['text'].values
        context = df['context'].values
        impact_label = df['impact_label'].values if self.is_labeling else None
        
        text_tokens_seq = list(map(self.tokenizer.tokenize, text))
        context_tokens_seq = list(map(self.tokenizer.tokenize, context))

        result = list(map(self.trunate_and_pad, context_tokens_seq, text_tokens_seq))
        input_ids = [i[0] for i in result]
        input_mask = [i[1] for i in result]
        segment_ids = [i[2] for i in result]
        
        if self.is_labeling:
            return torch.Tensor(input_ids).type(torch.long), \
                torch.Tensor(input_mask).type(torch.long), \
                torch.Tensor(segment_ids).type(torch.long), \
                torch.Tensor(impact_label).type(torch.long)   
        else:
            return torch.Tensor(input_ids).type(torch.long), \
                torch.Tensor(input_mask).type(torch.long), \
                torch.Tensor(segment_ids).type(torch.long), \
                impact_label
    
    def trunate_and_pad(self, tokens_a, tokens_b):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= self.max_seq_len - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
                
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        
        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        
        return input_ids, input_mask, segment_ids
    

if __name__ == "__main__":
    data_path = "C:/Users/31906/Desktop/HKUST/Curriculum/MSBD6000H Natural Language Processing/project1/data/train.csv"
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
    train_dataset  = AIC_Dataset(tokenizer, data_path, max_seq_len = 128, n_context=1, is_labeling=True)


