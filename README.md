# 安装
1. 首先需要cuda 11.7
2. 然后 conda 导入 environment.yml 创建环境 test
3. 切换到 test
# 运行
`python train pubmed pytorch test_eval.csv tsne_1 --cuda`
其中 PubMed 是数据集的名字,可以改成elliptic(bitcoin 数据集),reddit
pytorch 指的是调用 pytorch 进行训练,还有 tf 的,但是涉及的代码量太大,我只修改了 torch
test_eval.csv 表示输出的结果的名字
# 结果格式
结果由n行`kcore;0.6372866194056955;0.1461162567138672`构成
第一列表示算法名,第二次是 F1,第三列是训练时间
# 数据集依赖
在根目录下的datasets里面,创建相应的数据集的名字,在官网下载相应的数据集的文件,然后先执行一下train/dataname.py中的preprocess
reddit 数据集完全是胡扯, 作者甚至没有导入图的内容, 我自己改了一份能用的,不过单纯加载图就需要30 分钟以上.图的大小需要至少 20G 的显存和 30G 的内存
# 如何只运行部分算法?
在`__main__.py`中的 187 行开始
```python
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
```
可以将相应的train_timestep和evaluate注释掉即可




# Learning on Streaming Graphs with Experience Replay

This project applies experience replay to enable continuous graph representation learning in the streaming setting.
We update an Inductive Graph Neural Network while graph changes arrive as a stream or vertices or edges.

This repository contains the implementation of the following online training methods:
* Random-Based Rehearsal-RBR: yields a uniform sample of the entire training graph 
* Priority-Based Rehearsal-PBR: prioritizes datapoints based on the model prediction error

We also provide the following baseline implementations:
* No-rehearsal: trains over new vertices only
* Offline: performs multiple epochs over the entire graph


## Getting Started

* Clone the repository
* Install the dependencies
* Datasets:
  * Pubmed: https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz
  * Bitcoin: https://www.kaggle.com/ellipticco/elliptic-data-set
  * Reddit: http://snap.stanford.edu/graphsage/reddit.zip
  * Arxiv: https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
* Run the script (using python 3): 
```
python train <args>
```
or
```
python train/__main__.py <args>
```

### Prerequisites

Install the dependencies:

```
pip3 install -r requirements.txt
```

## Datasets
* [Pubmed](https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz) - Galileo Namata, et. al. "Query-driven Active Surveying for Collective Classification." MLG. 2012.
* [Reddit](http://snap.stanford.edu/graphsage/reddit.zip) - W.L. Hamilton et. al. "Inductive Representation Learning on Large Graphs", NeurIPS 2017
* [Elliptic Bitcoin](https://www.kaggle.com/ellipticco/elliptic-data-set) - Weber et. al. , "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", Anomaly Detection in Finance Workshop, 25th SIGKDD Conference on Knowledge Discovery and Data Mining
* [Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) - Weihua Hu et. al., "Open Graph Benchmark: Datasets for Machine Learning on Graphs." NeurIPS, 2020.
## Parameters
### Required
* ```dataset```: dataset: 'elliptic', 'pubmed', 'reddit' or 'tagged
* ```backend```: Deep learning framework. 'tf': Tensorflow (>= 2.2) or 'pytorch': Pytorch
* ```save_result```: output path
* ```save_tsne```: tsne plots path

### Optional
GPU
* ```--cuda```: Enable GPU accelleration
* ```--gpu <id>```: Use a specific GPU (CUDA must be enabled)

Model

* ```--latent_dim N```: size of the vector obtained applying a Pooling layer
* ```--embedding_size N```: size of the vector obtained applying a GraphSAGE aggregator (Pooling + Dense layers)
* ```--depth N```: sampling depth and number of GraphSAGE layers
* ```--samples N```: maximum number of neighbours sampled: in case of a vertex with degree >= of ```N```, N neighbours will be sampled
* ```--dropout N```: dropout used during the training phase

Train behaviour
* ```--batch_size N```: batch size used during the training phase (number of vertices trained together)
* ```--batch_full N```: batch size used during the evaluation/forward phase (number of vertices evaluated together)
* ```--snapshots N```: split the temporal graph into N snapshots
* ```--batch_timestep N```: train ```N``` batches in every snapshot 
* ```--eval N```: evaluate the model every ```N``` snapshots
* ```--epochs_offline N```: trains the offline model for ```N``` epochs
* ```--train_offline N```: trains the offline model every ```N``` snapshots
* ```--priority_forward N```: update the priorities running a forward pass every ```N``` snapshots
* ```--plot_tsne N```: generate a TSNE plot every N ```N``` snapshots
* ```--delta N```: evaluate the current model over a graph ```N``` snapshot in the future

### Example
```
python  train pubmed pytorch test_eval.csv tsne_1 --eval 10 --batch_timestep 20 --epochs_offline 25 --cuda
```
Runs the code using pytorch and the reddit pubmed. Results will be stored in test_eval.csv and TSNE plots in the tsne_1 folder.
The model is evaluated every 10 snapshots and 20 batches are trained in every timestep. The offline model is trained for 25 epochs. GPU accelleration is enabled.
