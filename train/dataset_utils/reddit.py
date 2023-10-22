from collections import defaultdict
import numpy as np
import pandas as pd
import os
import json
import networkx as nx
from networkx.readwrite import json_graph
import utils

from graph.dynamic_graph_vertex import DynamicGraphVertex
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

def load(path, snapshots=100, cuda=False, copy_to_gpu = False):
    # preprocess("./datasets/reddit")
    # a_exist = [f for f in FILES if os.path.isfile(os.path.join(path, f))]
    # if len(a_exist) < len(FILES):
    #     from dataset_utils.common_utils import downloadFromURL
    #     downloadFromURL(URL, path, True)

    feat_data = np.load(os.path.join(path, "feat_data.npy"))
    targets = np.load(os.path.join(path, "targets.npy"))
    targets = targets.astype(np.long)
    G = nx.read_adjlist(os.path.join(path, "graph.adjlist"))

    with open(os.path.join(path, 'vertex_timestamp.json')) as f:#last
        timestamps = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})

    g_dgl = dgl.from_networkx(G)
    g_dgl_test = dgl.from_networkx(G)

    feat_data_size = feat_data.shape[1]
    feat_data = utils.to_nn_lib(feat_data, cuda and copy_to_gpu)
    labelled_vertices = set(np.argwhere(targets != -1)[:, 0])
    n_classes =	len(np.unique(targets))
    targets = utils.to_nn_lib(targets, cuda and copy_to_gpu)

    g_dgl.ndata['feat'] = feat_data
    g_dgl.ndata['target'] = targets

    g_dgl_test.ndata['feat'] = feat_data
    g_dgl_test.ndata['target'] = targets

    dynamic_graph = DynamicGraphVertex(g_dgl, snapshots, labelled_vertices)
    dynamic_graph.build(vertex_timestamps=timestamps)

    dynamic_graph_test = DynamicGraphVertex(g_dgl_test, snapshots, labelled_vertices)
    dynamic_graph_test.build(vertex_timestamps=timestamps)

    return feat_data_size, targets, dynamic_graph, n_classes, dynamic_graph_test


if __name__ == "__main__":
    preprocess("../datasets/reddit")
