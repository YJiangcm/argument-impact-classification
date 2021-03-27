# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:21 2021

@author: JIANG Yuxin
"""


import torch.nn as nn


class AICModel(nn.Module):
    def __init__(self, model, n_label=3):
        super(AICModel, self).__init__()
        self.model = model #bert encoder
        for param in self.model.parameters():
            param.requires_grad = True  
        self.d_hid = model.config.hidden_size
        self.classifier = nn.Linear(self.d_hid, n_label)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        return logits, pooled_output
    
    
    