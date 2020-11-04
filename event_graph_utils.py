from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import networkx as nx
import random
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split


def mount_graph(df, path_to_language_model="../language_model"):
    print("Creating graph...")
    np.set_printoptions(threshold=100)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

    language_model = SentenceTransformer(path_to_language_model)

    df['embedding'] = list(language_model.encode(df['text'].to_list()))
    df = df.loc[~df['Themes'].isna()]

    G = nx.Graph()

    for index, row in df.iterrows():

        node_id = row['GKGRECORDID']
        node_date = str(row['DATE'])[0:8]
        node_themes_array = row['Themes'].split(';')
        node_locations_array = ''
        node_persons_array = ''
        node_organizations_array = ''

        try:
            node_locations_array = row['Locations'].split(';')
            node_persons_array = row['Persons'].split(';')
            node_organizations_array = row['Organizations'].split(';')
        except:
            1

        # event <-> date
        G.add_edge(node_id, node_date)
        G.nodes[node_id]['themes'] = node_themes_array
        G.nodes[node_date]['themes'] = node_themes_array

        # event <-> theme
        for theme in node_themes_array:
            if len(theme) > 0:
                G.add_edge(node_id, theme)
                G.nodes[theme]['themes'] = node_themes_array

        # event <-> locations
        for location in node_locations_array:
            if len(location) > 0:
                G.add_edge(node_id, location)
                G.nodes[location]['themes'] = node_themes_array

        # event <-> persons
        for person in node_persons_array:
            if len(person) > 0:
                G.add_edge(node_id, person)
                G.nodes[person]['themes'] = node_themes_array

        # event <-> organization
        for org in node_organizations_array:
            if len(org) > 0:
                G.add_edge(node_id, org)
                G.nodes[org]['themes'] = node_themes_array

        # embedding
        G.nodes[node_id]['embedding'] = row['embedding']

    # We'll relabel our nodes, since it's names are not convenient...
    mapping = {value: idx for idx, value in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping=mapping, copy=True)

    print(
        f"Graph loaded: OK - \t Nodes: {len(G.nodes)} \t  edges: {len(G.edges)}")
    return G


def regularization(G, dim, embedding_feature: str = 'embedding', iterations=15, mi=0.85):

    nodes = []

    # inicializando vetor f para todos os nodes
    for node in G.nodes():
        G.nodes[node]['f'] = np.array([0.0]*dim)
        if embedding_feature in G.nodes[node]:
            G.nodes[node]['f'] = G.nodes[node][embedding_feature]*1.0
        nodes.append(node)

    pbar = tqdm(range(0, iterations))

    for iteration in pbar:
        random.shuffle(nodes)
        energy = 0.0

        # percorrendo cada node
        for node in nodes:
            f_new = np.array([0.0]*dim)
            f_old = np.array(G.nodes[node]['f'])*1.0
            sum_w = 0.0

            # percorrendo vizinhos do onde
            for neighbor in G.neighbors(node):
                w = 1.0
                if 'weight' in G[node][neighbor]:
                    w = G[node][neighbor]['weight']

                w /= np.sqrt(G.degree[neighbor])

                f_new += w*G.nodes[neighbor]['f']

                sum_w += w

            f_new /= sum_w

            G.nodes[node]['f'] = f_new*1.0

            if embedding_feature in G.nodes[node]:
                G.nodes[node]['f'] = G.nodes[node][embedding_feature] * \
                    mi + G.nodes[node]['f']*(1.0-mi)

            energy += np.linalg.norm(f_new-f_old)

        iteration += 1
        message = 'Iteration '+str(iteration)+' | Energy = '+str(energy)
        pbar.set_description(message)

    return G


def process_event_dataset_from_networkx(G, features_attr="f", label_attr="themes", multi_label=False, train_split=0.6, val_split=0.2, random_state=42):
    """
    Builds an event graph dataset used in GAT model

    Parameters:
        G -> Graph representation of the event network (Networkx graph)
        features_att -> Feature attribute of each node (str)
        label_att -> Label attribute of each node (str)
        multi_label -> A boolean flag that considers a multi label dataset
        random_state -> A random seed to train_test_split
    Returns:
        adj -> Sparse and symmetric adjacency matrix of our graph.
        features -> A NumPy matrix with our graph features.
        labels -> A NumPy array with the labels of all nodes.
        idx_train -> A NumPy array with the indexes of the training nodes.
        idx_val -> A NumPy array with the indexes of the validation nodes.
        idx_test -> A NumPy array with the indexes of the test nodes.

    """

    num_nodes = len(G.nodes)
    single_themes_dict = {}

    idxs = np.arange(0, num_nodes)

    idx_train, idx_test_and_val = train_test_split(
        idxs, train_size=train_split, test_size=(1 - train_split), random_state=random_state)

    validation_split_percentage = val_split / (1 - train_split)

    idx_val, idx_test = train_test_split(
        idx_test_and_val, train_size=validation_split_percentage, random_state=random_state)

    # Organizing our feature matrix...
    # feature_matrix = np.array([ G.nodes[i]['embedding'] if 'embedding' in G.nodes[i].keys() else G.nodes[i][features_attr] for i in G.nodes()])
    features = np.array([G.nodes[i][features_attr] for i in G.nodes()])

    all_labels_from_all_nodes = nx.get_node_attributes(G, label_attr)

    # Adding our labels to our feature_matrix dataframe
    if (multi_label):
        labels = MultiLabelBinarizer().fit_transform(
            [tuple(all_labels_from_all_nodes[i]) for i in all_labels_from_all_nodes])
    else:
        labels = LabelEncoder().fit_transform(
            [all_labels_from_all_nodes[i][0] for i in all_labels_from_all_nodes])

    adj = nx.adjacency_matrix(G)

    return adj, features, labels, idx_train, idx_val, idx_test
