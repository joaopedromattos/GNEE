from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import new_load_data, accuracy
from models import GAT, SpGAT


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class GAT_wrapper():
    def __init__(self, args={"alpha": 0.2, "cuda": True, "dropout": 0.6, "epochs": 10, "fastmode": False, "hidden": 8, "lr": 0.005, "nb_heads": 8, "no_cuda": False, "patience": 100, "seed": 72, "sparse": False, "weight_decay": 0.0005}):

        if (type(args) == dict):
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
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        self.loss_test = loss_test
        self.acc_test = acc_test

        return loss_test, acc_test, output[self.idx_test].max(1)[1]

    def train_pipeline(self, *args, custom_function=True, function=None):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = new_load_data(
            *args, custom_function=custom_function, function=function)

        # Model and optimizer
        if self.args.sparse:
            model = SpGAT(nfeat=features.shape[1],
                          nhid=self.args.hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=self.args.dropout,
                          nheads=self.args.nb_heads,
                          alpha=self.args.alpha)
        else:
            model = GAT(nfeat=features.shape[1],
                        nhid=self.args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=self.args.dropout,
                        nheads=self.args.nb_heads,
                        alpha=self.args.alpha)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.args.lr,
                               weight_decay=self.args.weight_decay)

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
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            return loss_val.data.item()

        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = self.args.epochs + 1
        best_epoch = 0
        for epoch in range(self.args.epochs):
            loss_values.append(train(epoch))

            torch.save(model.state_dict(), '{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.args.patience:
                break

            files = glob.glob('*.pkl')
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Restore best model
        print('Loading {}th epoch'.format(best_epoch))
        model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

        self.model = model

        return model


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true',
                        default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true',
                        default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8,
                        help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8,
                        help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    gat = GAT_wrapper(args)
    gat.train_pipeline()
    gat.compute_test()
