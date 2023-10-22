from collections import defaultdict
import numpy as np
import pandas as pd
import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import utils

from graph.dynamic_graph_edge import DynamicGraphEdge
from ast import literal_eval as make_tuple

import dgl

#path = "datasets/reddit/"
FILES = ["feat_data.npy", "targets.npy", "edges_dataframe.csv"]
URL = "https://uoe-my.sharepoint.com/:u:/g/personal/s2121589_ed_ac_uk/EX6tn7RXc39LoIwZ0D5F9EcBofEGksT7nIuOrwIqCfXnPw?Download=1"

def preprocess(path, restrict=100000):
    id_map = json.load(open(os.path.join(path, "reddit-id_map.json")))

    G_data_json = json.load(open(os.path.join(path, "reddit-G.json")))
    G_data = json_graph.node_link_graph(G_data_json)
    G_data = G_data.to_undirected()

    nx.write_adjlist(G_data, os.path.join(path, "graph.adjlist"))
    feat_data = np.load(os.path.join(path, "reddit-feats.npy"))

    np.save(os.path.join(path, "feat_data.npy"), feat_data.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)

    labels = np.empty((feat_data.shape[0], 1), dtype=np.int64)
    targets_json = json.load(open(os.path.join(path, "reddit-class_map.json")))
    for k, v in targets_json.items():
        labels[id_map[k]] = int(v)

    np.save(os.path.join(path, "targets.npy"), feat_data.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)


def relabel():
    feat_data = np.load("feat_data.npy")
    targets = np.load("targets.npy")
    G = nx.read_adjlist("graph.adjlist", nodetype=int)
    with open('edge_timestamp.json') as f:
        timestamps = json.load(f, object_hook=lambda d: {tuple(map(int, make_tuple(k))): v for k, v in d.items()})

    def sort_second(val):
        return val[1]

    ordered_edges = list(timestamps.items())
    ordered_edges.sort(key=sort_second)
    edges = list(x[0] for x in ordered_edges)

    counter = 0
    vertices = {}

    for e in edges:
        v0 = e[0]
        v1 = e[1]

        if v0 not in vertices:
            vertices[v0] = counter
            counter += 1
        if v1 not in vertices:
            vertices[v1] = counter
            counter += 1

    new_graph = nx.relabel_nodes(G, vertices, copy=True)  # final graph
    new_timestamps = {}  # final timestamps
    for e in edges:
        new_timestamps[(vertices[e[0]], vertices[e[1]])] = timestamps[e]

    order = list(vertices.items())
    order.sort(key=sort_second)
    order = list(x[0] for x in order)

    feat_data_ordered = feat_data[order, :]
    targets_ordered = targets[order, :]

    str_new_timestamps = {}
    for x in new_timestamps.items():
        str_new_timestamps[str((x[0][0], x[0][1]))] = x[1]

    new_graph = new_graph.to_undirected()
    nx.write_adjlist(new_graph, "graph_relabel.adjlist")

    js = json.dumps(str_new_timestamps)
    f = open(("edge_relabel_timestamp.json"), "w")
    f.write(js)
    f.close()

    np.save(("feat_relabel_data.npy"), feat_data_ordered.astype(np.double, order='C'), allow_pickle=False,
            fix_imports=True)
    np.save(("targets_relabel.npy"), targets_ordered.astype(np.double, order='C'), allow_pickle=False, fix_imports=True)


def load(path, snapshots=100, cuda=False, copy_to_gpu = False):
    # preprocess("./datasets/reddit")
    # a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    # if len(a_exist) < len(FILES):
    #     from dataset_utils.common_utils import downloadFromURL
    #     downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feat_data.npy"))
    targets = np.load(os.path.join(path, "targets.npy"))
    targets = targets.astype(np.long)

    timestamps = pd.read_csv(os.path.join(path, "edges_dataframe.csv"), na_filter=False, dtype=np.int)

    print(feat_data.shape)
    print(len(timestamps))

    feat_data_size = feat_data.shape[1]
    #if cuda and copy_to_gpu:
    #    print("copy to gpu")
    feat_data = utils.to_nn_lib(feat_data, False)
    labelled_vertices = set(np.argwhere(targets != -1)[:,0])
    n_classes = len(np.unique(targets))
    targets = utils.to_nn_lib(targets, False)


    dynamic_graph = DynamicGraphEdge(snapshots, labelled_vertices)
    dynamic_graph.build(feat_data, targets, cuda and copy_to_gpu, edge_timestamps=timestamps)

    print("DONE")
    #dynamic_graph_test = None

    dynamic_graph_test = DynamicGraphEdge(snapshots, labelled_vertices)
    dynamic_graph_test.build(feat_data, targets, cuda and copy_to_gpu, edge_timestamps=timestamps)

    return feat_data_size, targets, dynamic_graph, n_classes, dynamic_graph_test


if __name__ == "__main__":
    preprocess("../datasets/reddit")
