# argument-impact-classification

This repository mainly illustrates my reimplementation and further research on the paper "[The Role of Pragmatic and Discourse Context in Determining Argument Impact](https://arxiv.org/pdf/2004.03034.pdf)" using PyTorch. If you want to get the data set, please contact the authors by email.

## Dataset
The dataset contains training, validation, and test data. Data are organized as 5 fields illustrated in Figure 2. The unique id is coupled with a training/validation/test example. The text is an argument. The context is a path from the root to the parent of text in the argument tree, and the stance label corresponds to the stances between two adjacent arguments in the path or the parent and the text (NULL is used to pad for the root). The impact label includes IMPACTFUL, MEDIUM IMPACT, NOT IMPACTFUL that need to be predicted. That is, this task is a 3-way classification problem evaluated by macro F1 score.

<img src="https://github.com/YJiangcm/argument-impact-classification/blob/master/picture/example.png" width="600" height="550">

## Experiments
I utilize PrLMs like BERT as well as other technics to do this task, and I surpass the best result of the original paper by around **6%**. Some of my experimental results are shown in the below table. 

## How to run
