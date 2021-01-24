# GNEE - GAT Neural Event Embeddings

This repository contains the implementation of the model Neural Event Embeddings Graph Attention Network (GNEE).

Our method consists of a pre-processment procedure followed by modified version of GAT
(Veličković et. al - 2017, https://arxiv.org/abs/1710.10903) to the event embedding task.

In out work, we adopt and modify the PyTorch implementation of GAT, [pyGAT](https://github.com/Diego999/pyGAT), developed by [Diego999](https://github.com/Diego999).


# File Structure
```
.
├── datasets_runs/ -> Datasets used
├── event_graph_utils.py -> Useful functions when working with event datasets
├── layers.py -> Implementation of Graph Attention layers
├── LICENSE
├── main.py -> Execute this script to reproduce our experiments (refer to our paper for more details)
├── models.py -> Implementation of the original GAT model
├── notebooks -> Run these notebooks to reproduce all our experiments.
├── README.md
├── requirements.txt
├── train.py -> Implementation of our preprocessing, traning and testing pipelines
└── utils.py -> Useful functions used in GAT original implementation.
```

# Performances

For the branch **master**, the training of the transductive learning on Cora task on a Titan Xp takes ~0.9 sec per epoch and 10-15 minutes for the whole training (~800 epochs). The final accuracy is between 84.2 and 85.3 (obtained on 5 different runs). For the branch **similar_impl_tensorflow**, the training takes less than 1 minute and reach ~83.0.

A small note about initial sparse matrix operations of https://github.com/tkipf/pygcn: they have been removed. Therefore, the current model take ~7GB on GRAM.

# Sparse version GAT

We develop a sparse version GAT using pytorch. There are numerically instability because of softmax function. Therefore, you need to initialize carefully. To use sparse version GAT, add flag `--sparse`. The performance of sparse version is similar with tensorflow. On a Titan Xp takes 0.08~0.14 sec.

# Requirements

pyGAT relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).

# Issues/Pull Requests/Feedbacks

