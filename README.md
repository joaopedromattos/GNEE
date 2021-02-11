# GNEE - GAT Neural Event Embeddings

This repository contains source code for the GNEE (GAT Neural Event Embeddings) method introduced in the paper: "Semi-Supervised Graph Attention Networks for Event Representation Learning".

Abstract: Event analysis from news and social networks is very useful for a wide range of social studies and real-world applications. Recently, event graphs have been explored to represent event datasets and their complex relationships, where events are vertices connected to other vertices that represent locations, people's names, dates, and various other event metadata. Graph representation learning methods are promising for extracting latent features from event graphs to enable the use of different classification algorithms. However, existing methods fail to meet important requirements for event graphs, such as (i) dealing with semi-supervised graph embedding to take advantage of some labeled events, (ii) automatically determining the importance of the relationships between event vertices and their metadata vertices, as well as (iii) dealing with the graph heterogeneity. In this paper, we present GNEE (GAT Neural Event Embeddings), a method that combines Graph Attention Networks and Graph Regularization. First, an event graph regularization is proposed to ensure that all graph vertices receive event features, thereby mitigating the graph heterogeneity drawback. Second, semi-supervised graph embedding with self-attention mechanism considers existing labeled events, as well as learns the importance of relationships in the event graph during the representation learning process. A statistical analysis of experimental results with five real-world event graphs and six graph embedding methods shows that GNEE obtains state-of-the-art results.

Our method consists of a BERT text encoding and a pre-processment procedure followed by modified version of GAT
(Veličković et. al - 2017, https://arxiv.org/abs/1710.10903) to the event embedding task.

In our work, we adopt and modify the PyTorch implementation of GAT, [pyGAT](https://github.com/Diego999/pyGAT), developed by [Diego999](https://github.com/Diego999).

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
