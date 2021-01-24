import torch
import argparse
import pandas as pd
from tqdm import tqdm
import networkx as nx
from os import listdir
from train import GAT_wrapper
from os.path import isfile, join
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from event_graph_utils import process_event_dataset_from_networkx, regularization

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="Validate during training pass.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="GAT with sparse version or not.",
    )
    parser.add_argument("--seed", type=int, default=72, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=10000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Initial learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
    parser.add_argument(
        "--nb_heads", type=int, default=8, help="Number of head attentions."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.6,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    parser.add_argument("--patience", type=int, default=100, help="Patience")

    # Parsing our args...
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    path_datasets = "datasets_runs/"
    network_files = [
        f for f in listdir(path_datasets) if isfile(join(path_datasets, f))
    ]

    experimental_results = []

    # Runs the full pipeline 10 times for all the datasets in dataset_runs/
    for i in range(1, 11):
        for network_file in tqdm(network_files):

            print("Networkfile", network_file)

            G = nx.read_gpickle(path_datasets + network_file)

            regularization(G, 512, embedding_feature="features")

            (
                adj,
                features,
                labels,
                idx_train,
                idx_test,
                df_labels,
            ) = process_event_dataset_from_networkx(G)

            print(adj.shape, features.shape, len(idx_train), len(idx_test))

            gat = GAT_wrapper(args)
            gat.train_pipeline(adj, features, labels, idx_train, idx_train, idx_test)

            loss, acc, output = gat.compute_test()

            y_pred = output.numpy()
            y_true = []

            for event_id in idx_test:
                for node in G.nodes():
                    if ":event" in node:
                        if G.nodes[node]["id"] == event_id:
                            y_true.append(
                                df_labels[
                                    df_labels.event_id == event_id
                                ].label_code.values[0]
                            )

            f1_macro = f1_score(y_true, y_pred, average="macro")
            acc = accuracy_score(y_true, y_pred)

            print("--->", network_file, "f1_macro", f1_macro, "acc", acc)
            experimental_results.append(
                (network_file, "f1_macro", f1_macro, "acc", acc, y_true, y_pred)
            )
            del gat
            del adj
            del features
            del G

        df_results = pd.DataFrame(experimental_results)
        df_results.to_csv(
            f"./gat_results_{i}_news_cluster_5w1h_graph_hin.csv", index=False
        )
