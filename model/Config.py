# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:21 2021

@author: JIANG Yuxin
"""


import argparse


def arg_conf():
    parser = argparse.ArgumentParser(description = 'Argument Imapct Classification')
    
    # parameters of environment
    parser.add_argument('-cuda', type=int, default=0, help="which device, default gpu.")
    parser.add_argument('-random_seed', type=int, default=2021, help='set the random seed so that we can reporduce the result.')
    
    # parameters of data processor
    parser.add_argument('-train_data_path', default=None)
    parser.add_argument('-dev_data_path', default=None)
    parser.add_argument('-test_data_path', default=None, help='data path of test dataset.')
    parser.add_argument('-n_label', type=int, default=3, help='number of labels.')
    parser.add_argument('-max_seq_len', type=int, default=128, help='max sequence length of input tokens.')
    parser.add_argument('-n_context', type=int, default=1, help='context input length.')
    
    # parameters of model
    parser.add_argument('-checkpoint', default=None, help='if use fine-tuned bert model, please enter the checkpoint path.')
    parser.add_argument('-bert_model', default=None, help='model name can be accessed from huggingface')
    
    # parameters of training
    parser.add_argument('-do_train', action='store_true', help='if training, default False')
    parser.add_argument('-do_eval', action='store_true', help='if evaluating, default False')
    parser.add_argument('-do_test', action='store_true', help='if testing, default False')
    parser.add_argument('-evaluate_steps', type=int, default=100, help='evaluate on the dev set at every xxx evaluate_steps.')
    parser.add_argument('-max_train_steps', type=int, default=-1, help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('-n_epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('-batch_size', type=int, default=16, help='number of examples per batch')
    parser.add_argument('-label_smoothing', type=float, default=0.0, help='the alpha parameter of label smoothing')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1, help='num of gradient_accumulation_steps') 
    parser.add_argument("-max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument('-weight_decay', type=float, default=0.01, help='regularize parameters')
    parser.add_argument('-lr', type=float, default=2e-5, help='initial learning rate')
    parser.add_argument('-save_path', default=None, help='model save path') 

    args = parser.parse_known_args()[0]
    
    print(vars(args))
    return args


if __name__ == "__main__":
    args = arg_conf()