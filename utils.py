import numpy as np
import scipy.sparse as sp
import torch
import spektral as spk
import pandas as pd
import networkx as nx
import os


def cora_networkx(path=None):
    if (path == None):
        raise ValueError("Dataset path shouldn't be of type 'None'.")
    else:
        # Reading our graph, according to documentation
        edgelist = pd.read_csv(os.path.join(
            path, "cora.cites"), sep='\t', header=None, names=["target", "source"])
        edgelist["label"] = "cites"

        # Transforming it into a
        Gnx = nx.from_pandas_edgelist(edgelist, edge_attr="label")
        nx.set_node_attributes(Gnx, "paper", "label")
        adj = nx.to_scipy_sparse_matrix(Gnx)

        # Sparse feature matrix
        feature_names = ["w_{}".format(ii) for ii in range(1433)]
        column_names = feature_names + ["subject"]
        node_data = pd.read_csv(os.path.join(
            path, "cora.content"), sep='\t', header=None, names=column_names)
        node_data.to_numpy()[:, :-1]
        features = sp.csr_matrix(node_data.to_numpy()[
                                 :, :-1], dtype=np.float32)

        # Train / val / test spliting...
        num_nodes = features.shape[0]
        idxs = np.arange(0, num_nodes)
        idx_train, idx_val, idx_test = np.split(
            idxs, [int(.6*num_nodes), int(.8*num_nodes)])

        labels = encode_onehot(node_data.to_numpy()[:, -1])

        return adj, features, labels, idx_train, idx_val, idx_test


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def new_load_data(path="./pyGAT/data/cora/", dataset='cora', use_networkx=True):
    print(f"[LOAD DATA]: {dataset}")

    if (not use_networkx):
        if (dataset == "cora" or dataset == 'citeseer' or dataset == 'pubmed'):
            adj, features, labels, train, val, test = spk.datasets.citation.load_data(
                dataset_name=dataset, normalize_features=True, random_split=True)
        elif (dataset == 'ppi' or dataset == 'reddit'):
            adj, features, labels, train, val, test = spk.datasets.graphsage.load_data(
                dataset_name=dataset, max_degree=1, normalize_features=True)
        else:
            raise ValueError(
                "Dataset not supported. List of supported datsets: ['cora', 'citeseer', 'pubmed', 'ppi', 'reddit']")
        print(f"ADJ {type(adj)}, \nFEATURES {type(features)}, \nLABELS {type(labels)}, \nTRAIN {type(train)}, \nVAL {type(val)}, \nTEST {type(test)}")

        # Converting one-hot encoding into categorical
        # values with the indexes of each dataset partition
        idx_train, idx_val, idx_test = np.where(train)[0], np.where(val)[
            0], np.where(test)[0]
    else:
        adj, features, labels, idx_train, idx_val, idx_test = cora_networkx(
            path)
    # Normalizing our features and adjacency matrices
    # features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(adj.todense())
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def original_load_data(path="./pyGAT/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Test {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
        labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalizing our features and adjacency matrices
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
