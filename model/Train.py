# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:21 2021

@author: JIANG Yuxin
"""


import os
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy import ndimage
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from Config import arg_conf
from Dataloader import AIC_Dataset
from Model import AICModel
from CrossEntropy import CrossEntropyLoss


def seed_torch(seed=2021):
    """set the random seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Metric(y_true, y_pred):
    """
    print the classification report
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    target_names = ['NOT_IMPACTFUL', 'MEDIUM_IMPACT', 'IMPACTFUL']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)

    print('Accuracy: {:.1%}\nPrecision: {:.1%}\nRecall: {:.1%}\nF1: {:.1%}'.format(accuracy, macro_precision,
                                           macro_recall, macro_f1))
    print("classification_report:\n")
    print(report)
    return macro_f1


def train(args, train_dataset, model, optimizer, eval_dataset, test_dataset):
    """ train the model """
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    if args.max_train_steps > 0:
        total_steps = args.max_train_steps
        args.n_epoch = args.max_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.n_epoch
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.n_epoch)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    global_step = 0
    tr_loss = 0.0
    best_eval_f1 = 0
    model.zero_grad()
    train_iterator = trange(int(args.n_epoch), desc="Epoch")
    
    if args.do_eval:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        output_eval_file = os.path.join(args.save_path, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("***** Eval results per %d training steps *****\n" % args.evaluate_steps)
    
    # added here for reproductibility
    seed_torch(args.random_seed)
    
    for epoch in train_iterator:
        
        epoch_iterator = tqdm(train_dataloader, desc="Training")
        batch_time_avg = 0.0
        labels = []
        predicts = []
        
        for step, batch in enumerate(epoch_iterator):
            batch_start = time.time()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            logits, _ = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                )
            impact_labels = batch[3]
            criterion = CrossEntropyLoss(weight=torch.Tensor([3.0, 3.0, 1.0]).to(args.device), smooth_eps=args.label_smoothing)
            loss = criterion(logits, impact_labels)
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            tr_loss += loss.item()
            old_global_step = global_step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                batch_time_avg += time.time() - batch_start
                description = "Avg. time per gradient updating: {:.4f}s, loss: {:.4f}"\
                                .format(batch_time_avg/(step+1), tr_loss/global_step)
                epoch_iterator.set_description(description)
        
            logits = logits.detach().cpu().numpy()
            impact_labels = impact_labels.detach().cpu().numpy()
            labels.extend(impact_labels)
            predicts.extend(np.argmax(logits, axis=-1))
            
            if args.do_eval:    
                if global_step != old_global_step and global_step % args.evaluate_steps == 0:
                    result = evaluate(args, eval_dataset, test_dataset, model, output_eval_file)
                    
                    # save the model having the best accuracy on dev dataset.
                    if result['eval_f1'] > best_eval_f1 and result['KL'] < 0.06:
                        best_eval_f1 = result['eval_f1']
                        now_time = time.strftime('%Y-%m-%d',time.localtime(time.time()))
                        torch.save({"model": model.state_dict(), 
                                    "name": args.bert_model, 
                                    "optimizer": optimizer.state_dict(), 
                                    },
                                    os.path.join(args.save_path, "model-" + now_time + ".pt"))
                        logger.info("***** Better eval f1, save model successfully *****")

            if args.max_train_steps > 0 and global_step > args.max_train_steps:
                epoch_iterator.close()
                break
        
        train_f1 = Metric(labels, predicts)
        logger.info("After epoch {:}, train_f1 = {:.2%}".format(epoch, train_f1))
        
        if args.max_train_steps > 0 and global_step > args.max_train_steps:
            epoch_iterator.close()
            break
        
    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, test_dataset, model, output_eval_file):
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    eval_loss = 0
    nb_eval_steps = 0
    labels = []
    predicts = []
    pooled_outputs = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            logits, pooled_output = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                )
            impact_labels = batch[3]
            criterion = CrossEntropyLoss(weight=torch.Tensor([3.0, 3.0, 1.0]).to(args.device), smooth_eps=args.label_smoothing)
            loss = criterion(logits, impact_labels)
            eval_loss += loss.item()
    
        logits = logits.detach().cpu().numpy()
        impact_labels = impact_labels.detach().cpu().numpy()
        labels.extend(impact_labels)
        predicts.extend(np.argmax(logits, axis=-1))
        pooled_outputs.extend(pooled_output.detach().cpu().numpy())
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    eval_f1 = Metric(labels, predicts)
    
    # Using PCA for visualization
    pca = PCA(n_components=2)
    pca.fit(pooled_outputs)
    logger.info("The explained_variance_ratio of PCA is: %s", pca.explained_variance_ratio_)
    low_dim_embs = pca.fit_transform(pooled_outputs)
    plt.figure(figsize=(20, 20))
    x = low_dim_embs[:, 0]
    y = low_dim_embs[:, 1]
    color =['not impactful' if l == 0 else 'medium_impactful' if l == 1 else 'impactful' for l in labels]
    df = pd.DataFrame(dict(x=x, y=y, color=color))
    sns.lmplot('x', 'y', data=df, hue='color', fit_reg=False)
    plt.savefig(os.path.join(args.save_path, "pca.jpg"))
    
    # Compute the KL-divergence between eval embeddings and test embeddings
    test_pooled_outputs = evaluate_test(args, test_dataset, model)
    KL = compute_kl(test_pooled_outputs, pooled_outputs)
    
    result = {"eval_loss": eval_loss, "eval_f1": eval_f1, "KL": KL}
    
    # write eval results to txt file.
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        logger.info("eval_loss = %.4f", eval_loss)
        logger.info("eval_f1 = {:.2%}".format(eval_f1))
        logger.info("KL-divergence between eval and test = %.4f", KL)
        
        writer.write("eval_loss = %s\n" % str(round(eval_loss, 4)))
        writer.write("eval_f1 = %s\n" % (str(round(eval_f1*100, 2))+'%'))
        writer.write("KL-divergence between eval and test = %s\n" % str(round(KL, 4)))

    return result


def kl(x, y, bins=100, EPS=1e-5, sigma=1):
    # histogram
    hist_xy = np.histogram2d(x, y, bins=bins)[0]

    # smooth it out for better results
    ndimage.gaussian_filter(hist_xy, sigma=sigma, mode='constant', output=hist_xy)

    # compute marginals
    hist_xy = hist_xy + EPS # prevent division with 0
    hist_xy = hist_xy / np.sum(hist_xy)
    hist_x = np.sum(hist_xy, axis=0)
    hist_y = np.sum(hist_xy, axis=1)

    kl = -np.sum(hist_x * np.log(hist_y / hist_x ))
    return kl


def compute_kl(eval_distribution, test_distribution):
    eval_distribution = np.array(eval_distribution).T
    test_distribution = np.array(test_distribution).T
    KL = 0.0
    for dim in range(len(eval_distribution)):
        KL += kl(test_distribution[dim], eval_distribution[dim])
    KL /= (dim+1)
    return KL
    
    
def evaluate_test(args, test_dataset, model):
    """ test the model """
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    
    # Test!
    logger.info("***** Evaluate test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    
    pooled_outputs = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            _, pooled_output = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                )
        pooled_outputs.extend(pooled_output.detach().cpu().numpy())
    
    return pooled_outputs
    
    

def test(args, test_dataset, model):
    """ test the model """
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    
    # Test!
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.batch_size)

    predictions = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            logits, _ = model(
                                    input_ids = batch[0], 
                                    attention_mask = batch[1],
                                    token_type_ids = batch[2],
                                )
    
        logits = logits.detach().cpu().numpy()
        prediction = np.argmax(logits, axis=-1)
        predictions.extend(prediction)
        
    # write predictions to csv file.
    ids = ['test_'+str(i) for i in range(len(predictions))]
    pd.DataFrame({"id": ids, "pred": predictions}).to_csv(os.path.join(args.save_path, "test_predictions.csv"), index= False)
    


if __name__ == "__main__":
    
    logger = logging.getLogger(__name__)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    args = arg_conf()
    seed_torch(args.random_seed)
    
    # if use GPU or CPU
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
        
    logger.info("  Device = %s", args.device)
        
    # data read and process
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    # tokenizer = DebertaTokenizer.from_pretrained(args.bert_model)
    
    if args.train_data_path:
        logger.info("***** Loading training data *****")
        train_dataset = AIC_Dataset(tokenizer, args.train_data_path, max_seq_len=args.max_seq_len, n_context=args.n_context, is_labeling=True)
            
    if args.dev_data_path:
        logger.info("***** Loading evaluating data *****")
        evaluate_dataset = AIC_Dataset(tokenizer, args.dev_data_path, max_seq_len=args.max_seq_len, n_context=args.n_context, is_labeling=True)
        
    if args.test_data_path:
        logger.info("***** Loading testing data *****")
        test_dataset = AIC_Dataset(tokenizer, args.test_data_path, max_seq_len=args.max_seq_len, n_context=args.n_context, is_labeling=False)
        
    # bulid the model 
    logger.info("***** Building model based on '%s' BERT model *****", args.bert_model)
    bert_model = AutoModel.from_pretrained(args.bert_model)
    # bert_model = DebertaModel.from_pretrained(args.bert_model)
    aic_model = AICModel(bert_model, args.n_label).to(args.device)
    # print the number of parameters of the model
    total_params = sum(p.numel() for p in aic_model.parameters())
    logger.info("{:,} total parameters.".format(total_params))
    total_trainable_params = sum(p.numel() for p in aic_model.parameters() if p.requires_grad)
    logger.info("{:,} training parameters.".format(total_trainable_params))

    # bulid the optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in aic_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    'weight_decay': args.weight_decay
            },
            {
                    'params':[p for n, p in aic_model.named_parameters() if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # load trained model from checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        if checkpoint["name"] == args.bert_model:
            logger.info("***** Loading saved model based on '%s' *****", checkpoint["name"])
            aic_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            raise Exception("The loaded model does not match the pre-trained model", checkpoint["name"])

    # train and evaluate
    if args.do_train == True and args.do_eval == True:
        global_step, tr_loss = train(args, train_dataset, aic_model, optimizer, evaluate_dataset, test_dataset)
        logger.info("***** End of training *****")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    # only evaluate
    elif args.do_train == False and args.do_eval == True:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        output_eval_file = os.path.join(args.save_path, "evaluate_result.txt")
        result = evaluate(args, evaluate_dataset, test_dataset, aic_model, output_eval_file)
        logger.info("***** End of evaluating *****")
    
    else:
        pass
    
    # test
    if args.do_test == True:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        test(args, test_dataset, aic_model)
        logger.info("***** End of testing *****")