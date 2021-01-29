import numpy as np
import networkx as nx
import random
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split


def regularization(
    G, dim, embedding_feature: str = "embedding", iterations=15, mi=0.85
):

    nodes = []

    # Initializing 'f' for all nodes...
    for node in G.nodes():
        G.nodes[node]["f"] = np.array([0.0] * dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]["f"] = G.nodes[node][embedding_feature] * 1.0
        nodes.append(node)

    pbar = tqdm(range(0, iterations))

    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0

        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0] * dim)
            f_old = np.array(G.nodes[node]["f"]) * 1.0
            sum_w = 0.0

            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if "weight" in G[node][neighbor]:
                    w = G[node][neighbor]["weight"]

                w /= np.sqrt(G.degree[neighbor])

                f_new += w * G.nodes[neighbor]["f"]

                sum_w += w

            f_new /= sum_w

            G.nodes[node]["f"] = f_new * 1.0

            if embedding_feature in G.nodes[node]:
                G.nodes[node]["f"] = G.nodes[node][embedding_feature] * mi + G.nodes[
                    node
                ]["f"] * (1.0 - mi)

            energy += np.linalg.norm(f_new - f_old)

        iteration += 1
        message = "Iteration " + str(iteration) + " | Energy = " + str(energy)
        pbar.set_description(message)

    return G


def process_event_dataset_from_networkx(G, features_attr="f"):
    """
    Builds an event graph dataset used in GAT model
    Parameters:
        G -> Graph representation of the event network (Networkx graph)
        df_labels -> user labeled data
        features_att -> Feature attribute of each node (str)
        random_state -> A random seed to train_test_split
    Returns:
        adj -> Sparse and symmetric adjacency matrix of our graph.
        features -> A NumPy matrix with our graph features.
        idx_train -> A NumPy array with the indexes of the training nodes.
        idx_val -> A NumPy array with the indexes of the validation nodes.
        idx_test -> A NumPy array with the indexes of the test nodes.
    """

    num_nodes = len(G.nodes)

    L_features = []
    L_train = []
    L_test = []
    L_labels = []
    label_codes = {}

    for node in G.nodes():

        L_features.append((G.nodes[node]["id"], G.nodes[node][features_attr]))

        if "train" in G.nodes[node]:
            L_train.append(G.nodes[node]["id"])

        if "test" in G.nodes[node]:
            L_test.append(G.nodes[node]["id"])

        if "label" in G.nodes[node]:

            if G.nodes[node]["label"] not in label_codes:
                label_codes[G.nodes[node]["label"]] = len(label_codes)

            L_labels.append(
                [
                    G.nodes[node]["id"],
                    G.nodes[node]["label"],
                    label_codes[G.nodes[node]["label"]],
                ]
            )

    df_features = pd.DataFrame(L_features)
    df_features.columns = ["node_id", "embedding"]

    features = np.array(df_features.sort_values(by=["node_id"])["embedding"].to_list())

    idx_train = L_train

    idx_test = L_test

    labels = [-1] * num_nodes
    df_labels = pd.DataFrame(L_labels)
    df_labels.columns = ["event_id", "label", "label_code"]

    for index, row in df_labels.iterrows():
        labels[row["event_id"]] = row["label_code"]

    adj = nx.adjacency_matrix(G)

    return adj, features, labels, idx_train, idx_test, df_labels
