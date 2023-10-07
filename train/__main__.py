import argparse
import json
import os
import gc
#import dgl
#import networkx as nx

from prioritized_replay.generate_priority import *

START_PRIOR_ALPHA = 4#10#1#0.3
END_PRIOR_ALPHA = 50#80#4#14
SCALE = 1

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=['elliptic', 'pubmed', 'reddit', 'tagged', 'flickr', 'arxiv'], help="Dataset")
parser.add_argument("backend", choices=['tf', 'pytorch', 'tf_static'], help="Framework (Tensorflow 2 or Pytorch)")
parser.add_argument("save_result", help="output file (.csv)")
parser.add_argument("save_tsne", help="path tsne plots")

parser.add_argument("--cuda", action='store_true', help="Enable CUDA")
parser.add_argument("--gpu", type=int, default=-1, help="Use a specific GPU (only if CUDA is enabled)")

parser.add_argument("--snapshots", type=int, help="split the temporal graph into N snapshots")
parser.add_argument("--embedding_size", type=int, help="intermediate latent size (between aggregation hops)")
parser.add_argument("--latent_dim", type=int, help="Pooling layers output size")
parser.add_argument("--depth", type=int, help="sampling depth")
parser.add_argument("--samples", type=int, help="neighbours sampled")
parser.add_argument("--batch_timestep", type=int, help="train N batches in every snapshot")
parser.add_argument("--eval", type=int, help="evaluate the model every N snapshots")
parser.add_argument("--batch_size", type=int, help="batch size used during the training phase")
parser.add_argument("--batch_full", type=int, help="batch size used during the testing phase")
parser.add_argument("--epochs_offline", type=int, help="trains the offline model for N epochs")
parser.add_argument("--train_offline", type=int, help="trains the offline model every N snapshots")
parser.add_argument("--priority_forward", type=int, help="update the priorities running a forward pass every N snapshots")
parser.add_argument("--plot_tsne", type=int, help="generate a TSNE plot every N snapshots")
parser.add_argument("--dropout", type=float, help="dropout used during the training phase")
parser.add_argument("--delta", type=int, help="evaluate the current model over a graph N snapshot in the future")

parser.add_argument("--n_sampling_workers", type=int, help="n. of parallel workers that sample the graph", default=0)
parser.add_argument("--copy_dataset_gpu", action='store_true', help="Copy the dataset to the GPU memory. Otherwise each batch will be copied")


args = parser.parse_args()
print(args)
custom_settings = {k: v for k, v in vars(args).items() if v is not None}

with open('settings/'+args.dataset+".json") as settings:
    data = json.load(settings)
    data.update(custom_settings)

if args.backend == "tf" or args.backend == "tf_static":
    os.environ['USE_OFFICIAL_TFDLPACK'] = "true"
    os.environ['DGLBACKEND'] = "tensorflow"
elif args.backend == "pytorch":
    os.environ['DGLBACKEND'] = "pytorch"

from utils import Lib_supported
if args.backend == "tf":
    LIBRARY = Lib_supported.TF
elif args.backend == "tf_static":
    LIBRARY = Lib_supported.TF_STATIC
elif args.backend == "pytorch":
    LIBRARY = Lib_supported.PYTORCH

import numpy as np
from graph.train_test_graph import TrainTestGraph
import random
from utils import init


import os, sys, time

class RedirIOStream:
    def __init__(self, stream, REDIRPATH):
        self.stream = stream
        self.path = REDIRPATH
    def write(self, data):
        # instead of actually writing, just append to file directly!
        myfile = open( self.path, 'a' )
        myfile.write(data)
        myfile.close()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


if not sys.stdout.isatty():
    # Detect redirected stdout and std error file locations!
    #  Warning: this will only work on LINUX machines
    STDOUTPATH = os.readlink('/proc/%d/fd/1' % os.getpid())
    STDERRPATH = os.readlink('/proc/%d/fd/2' % os.getpid())
    print("stdout path: ", STDOUTPATH)
    sys.stdout=RedirIOStream(sys.stdout, STDOUTPATH)
    sys.stderr=RedirIOStream(sys.stderr, STDERRPATH)



def run():
    print("init")
    GraphSAGE, RandomSupervisedGraphSage, PrioritizedSupervisedGraphSage, NoRehSupervisedGraphSage, FullSupervisedGraphSage, KcorePytorchSupervisedGraphSage, KcoreOneHopPytorchSupervisedGraphSage, KtrussNodePytorchSupervisedGraphSage, KtrussEdgePytorchSupervisedGraphSage, KtrussEdgeOneHopPytorchSupervisedGraphSage, activation = init(LIBRARY, data["cuda"], data["gpu"])
    if args.dataset == "pubmed":
        from dataset_utils.pubmed import load
    elif args.dataset == "reddit":
        from dataset_utils.reddit import load
    elif args.dataset == "elliptic":
        from dataset_utils.bitcoin import load
    elif args.dataset == "tagged":
        from dataset_utils.tagged import load
    elif args.dataset == "flickr":
        from dataset_utils.flickr import load
    elif args.dataset == "arxiv":
        from dataset_utils.arxiv import load

    print("load data")
    # graph 和 dynamic_graph_test 都是DynamicGraphVertex
    feat_data_size, labels, graph, n_classes, dynamic_graph_test = load(data["path"], snapshots=data["snapshots"], cuda=data["cuda"], copy_to_gpu = data["copy_dataset_gpu"])
    for i in range(data["delta"]):
        print("evolving delta")
        dynamic_graph_test.evolve()

    print("train test init")
    graph_util = TrainTestGraph(graph, split=0.15, start_prior_alpha=START_PRIOR_ALPHA, end_prior_alpha=END_PRIOR_ALPHA, scale=SCALE, max_priority=10)

    print("create graphsage")
    model_random = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_priority = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_no_reh = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_full = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_kcore = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_kcore_onehop = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_ktrussNode = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_ktrussEdge = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool", edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])
    model_ktrussEdge_onehop = GraphSAGE(feat_data_size, data["embedding_size"], n_classes, data["depth"] - 1, activation, data["dropout"], "pool",edge_feats=data["edge_feats"], pool_feats=data["latent_dim"])

    if data["cuda"] and LIBRARY == Lib_supported.PYTORCH:
        print("moving models to CUDA... ")
        model_random = model_random.cuda()
        model_priority = model_priority.cuda()
        model_no_reh = model_no_reh.cuda()
        model_full = model_full.cuda()
        model_kcore = model_kcore.cuda()
        model_kcore_onehop = model_kcore_onehop.cuda()
        model_ktrussNode = model_ktrussNode.cuda()
        model_ktrussEdge = model_ktrussEdge.cuda()
        model_ktrussEdge_onehop = model_ktrussEdge_onehop.cuda()

    print("start...")
    print(data["batch_timestep"]) # 这个值会被命令函参数修改成 20
    graphsage_random = RandomSupervisedGraphSage(model_random, data["batch_timestep"], data["batch_size"], labels, data["samples"],
                                                 cuda=data["cuda"], batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_random.build_optimizer()

    loss_priority = LossPriority()#labels.shape[0]

    graphsage_priority = PrioritizedSupervisedGraphSage(model_priority, data["batch_timestep"], data["batch_size"], labels, data["samples"], loss_priority,
                                                        cuda=data["cuda"], full_pass=data["priority_forward"], batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_priority.build_optimizer()

    graphsage_no_reh = NoRehSupervisedGraphSage(model_no_reh, data["batch_timestep"], data["batch_size"], labels, data["samples"],
                                                cuda=data["cuda"], batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_no_reh.build_optimizer()

    graphsage_full = FullSupervisedGraphSage(model_full, data["epochs_offline"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],
                                             batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_full.build_optimizer()

    graphsage_kcore = KcorePytorchSupervisedGraphSage(model_kcore, data["batch_timestep"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_kcore.build_optimizer()

    graphsage_kcore_onehop = KcoreOneHopPytorchSupervisedGraphSage(model_kcore_onehop, data["batch_timestep"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_kcore_onehop.build_optimizer()

    graphsage_ktrussNode = KtrussNodePytorchSupervisedGraphSage(model_ktrussNode, data["batch_timestep"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_ktrussNode.build_optimizer()

    graphsage_ktrussEdge = KtrussEdgePytorchSupervisedGraphSage(model_ktrussEdge, data["batch_timestep"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_ktrussEdge.build_optimizer()

    graphsage_ktrussEdge_onehop = KtrussEdgeOneHopPytorchSupervisedGraphSage(model_ktrussEdge_onehop, data["batch_timestep"], data["batch_size"], labels, data["samples"], cuda=data["cuda"],batch_full=data["batch_full"], n_workers = data["n_sampling_workers"])
    graphsage_ktrussEdge_onehop.build_optimizer()

    size_evolution = len(graph_util) #这是啥 分片的数量,graph_util可以不断地变成最新的
    print(size_evolution)

    import time


    for time_step in range(size_evolution):
        print("processing time step: ", time_step)

        #start = time.time()
        #print("random")
        graphsage_random.train_timestep(graph_util)
        #print("Time required: ", time.time() - start)
        #print("priority")
        graphsage_priority.train_timestep(graph_util)
        #print("no reh")
        graphsage_no_reh.train_timestep(graph_util)
        # print("kcore")
        graphsage_kcore.train_timestep(graph_util)
        # print("kcore one hop")
        graphsage_kcore_onehop.train_timestep(graph_util)
        # print("ktruss node")
        graphsage_ktrussNode.train_timestep(graph_util)
        # print("ktruss edge")
        graphsage_ktrussEdge.train_timestep(graph_util)
        # print("ktruss edge one hop")
        graphsage_ktrussEdge_onehop.train_timestep(graph_util)

        if time_step%data["train_offline"] == 0:
            print("train offline")
            graphsage_full.train_timestep(graph_util)

        if time_step % data["eval"] == 0:
            # 这里评估的是当前快照的结果
            # 每隔多少个时间戳就会调用这里
            graphsage_random.evaluate(graph_util, data["save_result"])
            graphsage_priority.evaluate(graph_util, data["save_result"])
            graphsage_no_reh.evaluate(graph_util, data["save_result"])
            graphsage_full.evaluate(graph_util, data["save_result"])
            graphsage_kcore.evaluate(graph_util, data["save_result"])
            graphsage_kcore_onehop.evaluate(graph_util, data["save_result"])
            graphsage_ktrussNode.evaluate(graph_util, data["save_result"])
            graphsage_ktrussEdge.evaluate(graph_util, data["save_result"])
            graphsage_ktrussEdge_onehop.evaluate(graph_util, data["save_result"])
            # 这里评估的是下一个快照的结果
            graphsage_random.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_priority.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_no_reh.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_full.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_kcore.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_kcore_onehop.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_ktrussNode.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_ktrussEdge.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])
            graphsage_ktrussEdge_onehop.evaluate_next_snapshots(dynamic_graph_test, data["delta"], data["save_result"])

        #if time_step % data["plot_tsne"] == 0:
        #    graphsage_priority.generate_tsne(graph_util, data["save_tsne"], time_step)


        if time_step + data["delta"] +1 < size_evolution:
            # 每个时间戳都会调用这里,这个函数会更新图
            print("evolving...")
            graph_util.evolve()
            dynamic_graph_test.evolve()
            gc.collect() # 垃圾回收????

        #if time_step % 50 == 0:
            #nxg = graph_util.temporal_graph.current_subgraph.to_networkx(node_attrs=['feat'])
            #graph_util.temporal_graph.current_subgraph = None
            #graph_util.graph = None
            #print("collected: ", gc.collect())
            #graph_util.temporal_graph.current_subgraph = dgl.DGLGraph()
            #graph_util.temporal_graph.current_subgraph.readonly(True)
            #graph_util.temporal_graph.current_subgraph.from_networkx(nxg, node_attrs=['feat'])
            #graph_util.temporal_graph.current_subgraph.readonly(True)
            #graph_util.graph = graph_util.temporal_graph.current_subgraph
        # break
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    np.random.seed(1)
    random.seed(1)
    run()
