#from graph.dynamic_graph import DynamicGraph
from sklearn.model_selection import train_test_split
import random
from itertools import compress
from prioritized_replay.replay_buffer import PrioritizedReplayBuffer
import utils
import networkx as nx
import dgl

#from memory_profiler import profile

SIZE_BUFFER = 10000000

class TrainTestGraph:
    '''
    A dynamic graph splitted in train/test
    '''

    def __init__(self, graph,split=0.25, start_prior_alpha=1, end_prior_alpha=2, scale=1, max_priority=3.0, start_priority = 2, min_priority=0.0000001):#TODO stratify
        '''

        :param graph: instance of DynamicGraph
        :param split: train-test split
        :param start_prior_alpha: start priority alpha value (coeff)
        :param end_prior_alpha: end priority alpha value (coeff)
        :param scale: increase in priority for each affected vertex
        :param max_priority: max allowed priority
        :param start_priority: new vertex priority
        :param min_priority: min. allowed priority
        '''
        self.scale = scale
        self.temporal_graph = graph
        self.train_set = set()
        self.test_set = set()
        self.size_evolution = len(graph)
        self.split = split
        self.graph = self.temporal_graph.get_graph()

        self.prior_alpha = start_prior_alpha

        self.start_prior_alpha = start_prior_alpha
        self.end_prior_alpha = end_prior_alpha

        self.max_priority = max_priority
        self.start_priority = start_priority
        self.min_priority = min_priority

        self.priority_replay_buffer = PrioritizedReplayBuffer(SIZE_BUFFER, self.prior_alpha, max_priority=self.max_priority, min_priority=self.min_priority) #up to 10 Million vertices
        added_vertices, labelled = self.temporal_graph.get_added_vertices() #new vertices
        labelled_vertices = list(compress(added_vertices, labelled))
        self._draw_train_test(list(labelled_vertices))

        self.core_change = []
        self.core_change_1hop = []
        self.nodeTruss_change = []
        self.edgeTruss_change = []
        self.edgeTruss_change_1hop = []


    def _draw_train_test(self, vertices):
        '''
        Split the new vertices into train and test
        :param vertices: the new vertices
        '''
        #print("draw train_test")
        if (len(vertices)>=3):
            self.train, self.test = train_test_split(vertices, shuffle=True, test_size=self.split)
        else:
            self.train = set(vertices)
            self.test = set()
        # train 是新增的节点, train_set 是所有的训练节点
        self.train_set = self.train_set.union(set(self.train))
        self.train_set_list = list(self.train_set)

        self.test_set = self.test_set.union(set(self.test))
        self.test_set_list = list(self.test_set)

        print("update priority")

        self._update_priority_struct() #update priority structure

        print("len train ", len(self.train_set_list))
        print("len test ", len(self.test_set_list))


    def _update_priority_struct(self):
        '''
        Updates priority tree after new vertices
        :return:
        '''

        n_vertices = {}
        min_val = self.priority_replay_buffer.get_min_priority()
        max_val = self.priority_replay_buffer.get_max_priority()

        #set priority for new vertices
        if self.priority_replay_buffer.get_max_priority() == -1:
            for vertex in self.train:
                n_vertices[vertex] = self.start_priority
        else:
            for vertex in self.train:
                n_vertices[vertex] = min_val + (max_val - min_val)*0.95

        self.priority_replay_buffer.add_all(n_vertices)

        #get vertices affected by changes
        '''
        dicts = {}
        vertices_affected_by_changes, depth = self.temporal_graph.get_vertices_changed()
        vertices_affected_by_changes = vertices_affected_by_changes.intersection(self.train_set)#vertices in the training set

        #convert vertices_affected_by_changes
        #print("Affected by changes: ",vertices_affected_by_changes)

        l_vertices = self.temporal_graph.get_labelled_vertices().intersection(vertices_affected_by_changes)

        #conversion

        original_to_subgraph = self.temporal_graph.get_original_to_subgraph_map()
        vertices_affected_by_changes = set(original_to_subgraph[list(vertices_affected_by_changes)])
        l_vertices = set(original_to_subgraph[list(l_vertices)])

        '''
        '''
        for vertex in vertices_affected_by_changes:
            d_vertices = self._get_affected_nodes(vertex, depth=depth)

            for key, value in d_vertices.items():

                if key not in l_vertices:#if a vertex affected is not labelled -> ignore
                    continue

                if key in dicts: #vertex influenced by multiple updates
                    dicts[key] = min(1, value+dicts[key])#TODO fix
                else:
                    dicts[key] = value
        

        subgraph_to_original = self.temporal_graph.get_subgraph_to_original_map()
        for k, v in dicts.items():
            k = subgraph_to_original[k]#conversion
            if k in self.train_set and k not in self.train:
                self.priority_replay_buffer.increment_priorities(k, v*0.5)#increment priorities
        '''


    def _get_affected_nodes(self, source_node, depth=2):
        '''
        :param source_node: new node (starting node)
        :param depth: depth of search
        :return: dict {node:priority update}
        '''

        nbrs = {source_node:1*self.scale}

        for l in range(depth):
            nbrs_tmp = {}
            for k, v in nbrs.items():
                for nbr in self.graph.predecessors(k):#k: "original" value -> subgraph value
                    nbr = utils.from_nn_get_python_value(nbr)
                    if nbr in nbrs_tmp:
                        nbrs_tmp[nbr] += (1 / (self.graph.out_degree(nbr))) * v * self.scale
                        nbrs_tmp[nbr] = min(nbrs_tmp[nbr], 1)
                    else:
                        nbrs_tmp[nbr] = (1/(self.graph.out_degree(nbr))) * v * self.scale

            #merge dicts
            for k, v in nbrs_tmp.items():
                if k in nbrs:
                    nbrs[k] = max(nbrs[k], v)
                else:
                    nbrs[k] = v

        return nbrs

    def __len__(self):
        return len(self.temporal_graph)

    def solveTruss(self, graph):
        truss1 = dict()
        # 计算此时graph中的每条边的k-truss
        # 遍历每条边,计算支持度
        for edge in graph.edges():
            # 边的两个端点
            u = edge[0]
            v = edge[1]
            # u的邻居
            u_neighbors = nx.neighbors(graph, u)
            # v的邻居
            v_neighbors = nx.neighbors(graph, v)
            # u和v的公共邻居
            common_neighbors = set(u_neighbors).intersection(set(v_neighbors))
            # print("common_neighbors: ", common_neighbors)
            # 计算支持度
            support = len(common_neighbors)
            # 输出支持度
            # print("support: ", support)
            if u not in truss1:
                truss1[u] = support
            else:
                truss1[u] = max(truss1[u], support)
            if v not in truss1:
                truss1[v] = support
            else:
                truss1[v] = max(truss1[v], support)
        # print("truss1: ", truss1) 
        return truss1
    def solveTrussEdge(self, graph):
        truss1 = dict()
        # 计算此时graph中的每条边的k-truss
        # 遍历每条边,计算支持度
        for edge in graph.edges():
            # 边的两个端点
            u = edge[0]
            v = edge[1]
            # u的邻居
            u_neighbors = nx.neighbors(graph, u)
            # v的邻居
            v_neighbors = nx.neighbors(graph, v)
            # u和v的公共邻居
            common_neighbors = set(u_neighbors).intersection(set(v_neighbors))
            # print("common_neighbors: ", common_neighbors)
            # 计算支持度
            support = len(common_neighbors)
            # 输出支持度
            # print("support: ", support)
            truss1[edge] = support
        return truss1

    #@profile
    def evolve(self):
        '''
        evolve the graph
        :return:
        '''
        self.prior_alpha = self.start_prior_alpha + (((self.end_prior_alpha - self.start_prior_alpha)/self.__len__()) * self.temporal_graph.evolution_index)

        # print("now nodes: ", self.graph.number_of_nodes())
        # 计算此时graph中的 kcore
        # 刚开始是edge 0 是因为没有边!
        # print("now nodes: ", self.graph.number_of_nodes(), self.graph.number_of_edges())
        nx_g = self.graph.to_networkx()
        nx_g = nx.Graph(nx_g)
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
        # print("nx_g: ", nx_g.number_of_nodes(), nx_g.number_of_edges()) 
        core1 = nx.algorithms.core.core_number(nx_g)
        # print("core1: ", core1)
        
        # 计算此时graph中的每个点的最大k-truss
        nodeTruss1 = self.solveTruss(nx_g)
        edgeTruss1 = self.solveTrussEdge(nx_g)

        # 更新图
        self.temporal_graph.evolve()
        self.graph = self.temporal_graph.get_graph()
        nx_g = self.graph.to_networkx()
        nx_g = nx.Graph(nx_g)

        # 计算此时graph中的 kcore
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
        core2 = nx.algorithms.core.core_number(nx_g)
        # print("core2: ", core2)
        # 计算此时graph中的每个点的最大k-truss
        nodeTruss2 = self.solveTruss(nx_g)
        edgeTruss2 = self.solveTrussEdge(nx_g)

        # 把两次出现变化的记录下来,结合新增的节点,得到需要训练的节点
        # 计算core1中出现的节点里 core 值变化的节点
        core_change = []
        for node in core1:
            if core1[node] != core2[node]:
                core_change.append(node)
        self.core_change = core_change
        
        # 计算truss1中出现的节点里 truss 值变化的节点
        nodeTruss_change = []
        for node in nodeTruss1:
            if nodeTruss1[node] != nodeTruss2[node]:
                nodeTruss_change.append(node)
        self.nodeTruss_change = nodeTruss_change

        # 计算edgeTruss1中出现的边里 truss 值变化的边
        edgeTruss_change = []
        for edge in edgeTruss1:
            if edgeTruss1[edge] != edgeTruss2[edge]:
                # 将边的两个端点加入到edgeTruss_change中
                edgeTruss_change.append(edge[0])
                edgeTruss_change.append(edge[1])
        self.edgeTruss_change = edgeTruss_change
        # print("edgeTrussChange: ", self.edgeTruss_change)
        

        core_change_1hop = []
        for node in core_change:
            core_change_1hop.extend(nx.neighbors(nx_g, node))
        edgeTruss_change_1hop = []
        for node in edgeTruss_change:
            edgeTruss_change_1hop.extend(nx.neighbors(nx_g, node))
        # print("core_change: ", core_change)
        # print("core_change_1hop: ", core_change_1hop)
            
        # print("core_change: ", len(self.core_change))

        # print("now nodes: ", self.graph.number_of_nodes())
        added_vertices, labelled = self.temporal_graph.get_added_vertices()
        # print("added vertices: ", len(added_vertices)) 这是新增的节点
        labelled_vertices = list(compress(added_vertices, labelled))
        self._draw_train_test(labelled_vertices)

    def get_graph(self):
        return self.temporal_graph.get_graph()


    def get_train_set(self):
        return self.train_set_list

    def get_test_set(self):
        return self.test_set_list

    def get_new_train_nodes(self, batch_size=None):
        # print("rel_train",self.train)
        l_train = list(self.train)
        if batch_size is None:
            return l_train

        if batch_size>=len(l_train):
            return l_train

        random.shuffle(l_train)
        return l_train[:batch_size]

    def get_new_test_nodes(self):
        return self.test

    def draw_random_train_nodes(self, n_nodes):
        #random.shuffle(self.train_set_list)
        if n_nodes <= len(self.train_set_list):
            random.shuffle(self.train_set_list)
            return self.train_set_list[:n_nodes]
        #print("list: ", self.train_set_list)
        return self.train_set_list #random.choices(self.train_set_list, k=n_nodes)

    def draw_priority_train_nodes(self, n_nodes):
        if n_nodes <= len(self.train_set_list):
            random.shuffle(self.train_set_list)
            return self.train_set_list[:n_nodes]

        return self.priority_replay_buffer.sample(n_nodes)

    def dump_priorities(self, vertex_list):
        return self.priority_replay_buffer.dump_priorities(vertex_list)

    def update_priorities(self, d_priorities):
        '''
        Sets the new priorities
        :param d_priorities: dict(node id, priority value)
        :return:
        '''

        assert(len(d_priorities) <= len(self.train_set))

        if len(d_priorities) < len(self.train_set):
            self.priority_replay_buffer.update_priorities(d_priorities)
        else:
            print("rebuild index, got ", len(d_priorities), " vertices")
            self.priority_replay_buffer = PrioritizedReplayBuffer(SIZE_BUFFER, self.prior_alpha, max_priority=self.max_priority, min_priority=self.min_priority)
            self.priority_replay_buffer.add_all(d_priorities)

    def get_original_to_subgraph_map(self):
        return self.temporal_graph.get_original_to_subgraph_map()

    def get_subgraph_to_original_map(self):
        return self.temporal_graph.get_subgraph_to_original_map()
