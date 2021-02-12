from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp

from sklearn.manifold import TSNE

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from utils import accuracy, normalize_adj
from models import GAT, SpGAT


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class GAT_wrapper:
    def __init__(
        self,
        args={
            "alpha": 0.2,
            "cuda": True,
            "dropout": 0.6,
            "epochs": 10,
            "fastmode": False,
            "hidden": 8,
            "lr": 0.005,
            "nb_heads": 8,
            "no_cuda": False,
            "patience": 100,
            "seed": 72,
            "sparse": False,
            "weight_decay": 0.0005,
        },
    ):

        if type(args) == dict:
            args = Namespace(args)

        self.args = args

        self.model = None

        self.loss_test = 0.0
        self.acc_test = 0.0

        self.adj = None
        self.features = None
        self.labels = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

    def compute_test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_test = F.nll_loss(
            output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print(
            "Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
        )

        self.loss_test = loss_test
        self.acc_test = acc_test

        return loss_test, acc_test, output[self.idx_test].max(1)[1]

    def train_pipeline(
        self, adj, features, labels, idx_train, idx_val, idx_test, *args
    ):

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        if sp.issparse(adj):
            adj = adj.todense()

        if sp.issparse(features):
            features = features.todense()

        # With networkx, we no longer need to convert from one-hot encoding...
        # labels = np.where(labels)[1]

        adj = torch.FloatTensor(adj)
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # Model and optimizer
        if self.args.sparse:
            model = SpGAT(
                nfeat=features.shape[1],
                nhid=self.args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=self.args.dropout,
                nheads=self.args.nb_heads,
                alpha=self.args.alpha,
            )
        else:
            model = GAT(
                nfeat=features.shape[1],
                nhid=self.args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=self.args.dropout,
                nheads=self.args.nb_heads,
                alpha=self.args.alpha,
            )
        optimizer = optim.Adam(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        if self.args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        features, adj, labels = Variable(
            features), Variable(adj), Variable(labels)

        # TODO: Test if these lines could be written below line 41.
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not self.args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            print(
                "Epoch: {:04d}".format(epoch + 1),
                "loss_train: {:.4f}".format(loss_train.data.item()),
                "acc_train: {:.4f}".format(acc_train.data.item()),
                "loss_val: {:.4f}".format(loss_val.data.item()),
                "acc_val: {:.4f}".format(acc_val.data.item()),
                "time: {:.4f}s".format(time.time() - t),
            )

            return loss_val.data.item()

        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = self.args.epochs + 1
        best_epoch = 0
        for epoch in range(self.args.epochs):
            loss_values.append(train(epoch))

            torch.save(model.state_dict(), "{}.pkl".format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.args.patience:
                break

            files = glob.glob("*.pkl")
            for file in files:
                epoch_nb = int(file.split(".")[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob("*.pkl")
        for file in files:
            epoch_nb = int(file.split(".")[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print("Loading {}th epoch".format(best_epoch))
        model.load_state_dict(torch.load("{}.pkl".format(best_epoch)))

        self.model = model

        return model

    def visualize_embeddings(self, G):

        embedding = self.model.get_attention_heads_outputs(
            self.features, self.adj).detach().numpy()

        counter = 0
        X = []
        Y = []

        for node in G.nodes():
            if ':event' in node:
                X.append(embedding[counter])
                Y.append(G.nodes[node]['label'])

            counter += 1

        X = np.array(X)

        X_embedded = TSNE(n_components=2).fit_transform(X)
        X_embedded.shape

        df = pd.DataFrame(X_embedded)
        df['label'] = Y
        df = df.dropna()

        g = sns.scatterplot(x=0, y=1, data=df, hue="label", legend=False)
        g.set(xlabel=None)
        g.set(ylabel=None)

        plt.savefig('../gnee.pdf')
