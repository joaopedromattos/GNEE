# GNEE - GAT Neural Event Embeddings

This repository contains the implementation of the model Neural Event Embeddings Graph Attention Network (GNEE).

Our method consists of a BERT text encoding and a pre-processment procedure followed by modified version of GAT
(Veličković et. al - 2017, https://arxiv.org/abs/1710.10903) to the event embedding task.

In out work, we adopt and modify the PyTorch implementation of GAT, [pyGAT](https://github.com/Diego999/pyGAT), developed by [Diego999](https://github.com/Diego999).

# Hardware requirements

When running on "dense" mode (no ```--sparse``` flag), our model uses about 18 GB on GRAM. On the other hand, the sparse mode (using ```--sparse```) uses less than 1.5 GB on GRAM, which is an ideal setup to environments such as Google Colab.


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

# Issues/Pull Requests/Feedbacks

Please, contact the authors in case of issues / pull requests / feedbacks :)