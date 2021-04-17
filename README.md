# argument-impact-classification

This repository contains the details about our course MSBD6000H project: [Argument Impact Classification](https://www.kaggle.com/c/ml4nlp-argimpact). 

Team name: **Bayes is all you need**  
Macro F1 score on the leaderboard: **62.439%**   
Rank: **2**

You could find our system description paper in the "report.pdf" file. All the code were written in PyTorch. 

## 1. Dataset
The dataset comes from the paper "[The Role of Pragmatic and Discourse Context in Determining Argument Impact](https://arxiv.org/pdf/2004.03034.pdf)". It contains training, validation, and test data. Data are organized as 5 fields illustrated in Figure 2. The unique id is coupled with a training/validation/test example. The text is an argument. The context is a path from the root to the parent of text in the argument tree, and the stance label corresponds to the stances between two adjacent arguments in the path or the parent and the text (NULL is used to pad for the root). The impact label includes IMPACTFUL, MEDIUM IMPACT, NOT IMPACTFUL that need to be predicted. That is, this task is a 3-way classification problem evaluated by macro F1 score.

<img src="https://github.com/YJiangcm/argument-impact-classification/blob/master/picture/example.png" width="600" height="550">

## 2. Experiments
I utilize PrLMs like BERT as well as other technics to do this task, and I surpass the best result of the original paper by around **6%**. Some of my experimental results are shown in the below table. 

<img src="https://github.com/YJiangcm/argument-impact-classification/blob/master/picture/experiments.png" width="600" height="350">

## 3. How to run
I have add all the code to this repository. It is recommanded to run the code in **Google Colab**, where you could acquire free GPU resources. Or you can just simply run my code in your terminal like this:

### 3.1 Model training and evaluating
```
! python Train.py -train_data_path='/content/drive/My Drive/argument impact classification/data/train.csv' \
          -dev_data_path='/content/drive/My Drive/argument impact classification/data/valid.csv' \
          -n_label=3 \
          -max_seq_len=384 \
          -n_context=5 \
          -bert_model='bert-base-uncased' \
          -do_train \
          -do_eval \
          -evaluate_steps=100 \
          -max_train_steps=-1 \
          -n_epoch=10 \
          -batch_size=16 \
          -label_smoothing=0.1 \
          -gradient_accumulation_steps=1 \
          -max_grad_norm=1.0 \
          -weight_decay=0.05 \
          -lr=2e-5 \
          -save_path='/content/drive/My Drive/argument impact classification/log/bert-base-C5'
```

### 3.2 Model predicting
```
! python Train.py -test_data_path='/content/drive/My Drive/argument impact classification/data/test.csv' \
          -n_label=3 \
          -max_seq_len=384 \
          -n_context=5 \
          -checkpoint='/content/drive/My Drive/argument impact classification/log/bert-base-C5/model-2021-03-26.pt' \
          -bert_model='bert-base-uncased' \
          -do_test \
          -batch_size=16 \
          -save_path='/content/drive/My Drive/argument impact classification/log/bert-base-C5'
```


## LICENSE
If you find this repository helps you, do not hesitate to give me a star ⭐. For copyright, please refer to [MIT License Copyright (c) 2021 YJiangcm](https://github.com/YJiangcm/argument-impact-classification/blob/master/LICENSE).
