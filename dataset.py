import json
import math
import random
import torch
import numpy as np
import pickle as pkl
import torch.nn as nn
import networkx as nx
import torch.nn.functional as F
from numpy.random import RandomState
from collections import defaultdict
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class Dataset:
    def __init__(self, name, args):
        self.dataset_name = name
        self.args = args
        self.train_graphs = []
        self.test_graphs = []

        # print(self.dataset_name)

        all_graphs, label_dict, tagset = load_data(self.dataset_name, True)
        # all_classes = list(label_dict.keys())

        self.tagset = tagset

        with open("../split/{}/main_splits.json".format(args.dataset_name), "r") as f:
            all_class_splits = json.load(f)
            self.train_classes = all_class_splits["train"]
            self.test_classes = all_class_splits["test"]

        train_classes_mapping = {}
        for cl in self.train_classes:
            train_classes_mapping[cl] = len(train_classes_mapping)
        self.train_classes_num = len(train_classes_mapping)

        test_classes_mapping = {}
        for cl in self.test_classes:
            test_classes_mapping[cl] = len(test_classes_mapping)
        self.test_classes_num = len(test_classes_mapping)

        for i in range(len(all_graphs)):
            if all_graphs[i].label in self.train_classes:
                self.train_graphs.append(all_graphs[i])

            if all_graphs[i].label in self.test_classes:
                self.test_graphs.append(all_graphs[i])

        for graph in self.train_graphs:
            graph.label = train_classes_mapping[int(graph.label)]
        for i, graph in enumerate(self.test_graphs):
            graph.label = test_classes_mapping[int(graph.label)]

        #
        # num_validation_graphs = math.floor(0.2 * len(self.train_graphs))
        # num_validation_graphs = math.floor(0 * len(self.train_graphs))

        # np.random.seed(seed_value)
        np.random.shuffle(self.train_graphs)

        # self.train_graphs = self.train_graphs[: len(self.train_graphs) - num_validation_graphs]
        # self.validation_graphs = self.train_graphs[len(self.train_graphs) - num_validation_graphs:]

        self.train_tasks = defaultdict(list)
        for graph in self.train_graphs:
            self.train_tasks[graph.label].append(graph)

        # perform mix-up for train graphs
        generate_train_graphs = []
        fir, sec = np.random.randint(low=0, high=len(self.train_graphs), size=(2, self.args.gen_train_num))
        for i in range(self.args.gen_train_num):
            if i % 64 == 0:
                lam = np.random.beta(0.5, 0.5)
            lam = max(lam, 1 - lam)
            # emb1 = self.train_graphs[fir[i]].node_features / self.train_graphs[fir[i]].norm(dim=1)
            match = self.train_graphs[fir[i]].node_features @ self.train_graphs[sec[i]].node_features.T
            normalized_match = F.softmax(match, dim=0)

            mixed_adj = lam * to_dense_adj(self.train_graphs[fir[i]].edge_mat)[0].double() + (
                        1 - lam) * normalized_match.double() @ to_dense_adj(self.train_graphs[sec[i]].edge_mat)[
                            0].double() @ normalized_match.double().T
            mixed_adj[mixed_adj < 0.1] = 0
            mixed_x = lam * self.train_graphs[fir[i]].node_features + (1 - lam) * normalized_match.float() @ \
                      self.train_graphs[sec[i]].node_features

            edge_index, _ = dense_to_sparse(mixed_adj)
            edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
            g = nx.Graph()
            g.add_edges_from(edges)
            g.add_nodes_from(list(range(edge_index.max() + 1)))
            G = Graph(g, -1)
            G.edge_mat = edge_index
            G.node_features = mixed_x
            generate_train_graphs.append(G)
        print("generate yes")
        print("generate len is ", len(generate_train_graphs))
        print("before the number of train graphs is ", len(self.train_graphs))
        self.train_graphs.extend(generate_train_graphs)
        print("after the number of train graphs is ", len(self.train_graphs))

        #         self.valid_tasks = defaultdict(list)
        #         for graph in self.validation_graphs:
        #             self.valid_tasks[graph.label].append(graph)

        # np.random.seed(2)
        # np.random.seed(seed_value)
        np.random.shuffle(self.test_graphs)

        # self.test_graphs = self.test_graphs[:self.args.K_shot + self.args.N_way * (self.args.query_size) * 200] # no useful

        # nx.Graph().number_of_nodes()

        self.test_tasks = defaultdict(list)
        for graph in self.test_graphs:
            self.test_tasks[graph.label].append(graph)

        self.test_fine_tune_list = []
        # self.test_fine_tune = defaultdict(list)
        self.total_test_g_list = []
        for index in range(self.test_classes_num):
            self.test_fine_tune_list.append(list(self.test_tasks[index])[:self.args.K_shot])
            # self.test_fine_tune[index].extend(list(self.test_tasks[index])[:self.args.K_shot])
            self.total_test_g_list.extend(list(self.test_tasks[index])[self.args.K_shot:])

        # perform mixup for test graphs
        self.generate_test_graphs = defaultdict(list)
        for index in range(self.test_classes_num):
            fir, sec = np.random.randint(low=0, high=len(self.test_fine_tune_list), size=(2, self.args.gen_test_num))
            for i in range(self.args.gen_test_num):
                lam = np.random.beta(0.5, 0.5)
                lam = max(lam, 1 - lam)

                # emb1 = self.train_graphs[fir[i]].node_features / self.train_graphs[fir[i]].norm(dim=1)
                match = self.test_fine_tune_list[index][fir[i]].node_features @ self.test_fine_tune_list[index][
                    sec[i]].node_features.T
                normalized_match = F.softmax(match, dim=0)
                mixed_adj = lam * to_dense_adj(self.test_fine_tune_list[index][fir[i]].edge_mat)[0].double() + (
                            1 - lam) * normalized_match.double() @ \
                            to_dense_adj(self.test_fine_tune_list[index][sec[i]].edge_mat)[
                                0].double() @ normalized_match.double().T
                mixed_adj[mixed_adj < 0.1] = 0
                mixed_x = lam * self.test_fine_tune_list[index][fir[i]].node_features + (
                            1 - lam) * normalized_match.float() @ self.test_fine_tune_list[index][sec[i]].node_features
                # mixed_y = [self.test_fine_tune_list[fir[i]].label, self.test_fine_tune_list[sec[i]].label]

                edge_index, _ = dense_to_sparse(mixed_adj)
                edges = [(x, y) for x, y in zip(edge_index[0].tolist(), edge_index[1].tolist())]
                g = nx.Graph()
                g.add_edges_from(edges)
                g.add_nodes_from(list(range(edge_index.max() + 1)))
                G = Graph(g, -2)
                G.edge_mat = edge_index
                G.node_features = mixed_x
                G.y_a = self.test_fine_tune_list[index][fir[i]].label
                G.y_b = self.test_fine_tune_list[index][sec[i]].label
                G.lam = lam
                self.generate_test_graphs[index].append(G)

        print("generate yes")
        #         print("generate len is ", len(generate_test_graphs))
        #         print("before the number of train graphs is ", len(self.test_graphs))
        #         self.train_graphs.extend(generate_train_graphs)
        #         print("after the number of train graphs is ", len(self.train_graphs))
        rd = RandomState(0)
        rd.shuffle(self.total_test_g_list)
        # rd.shuffle(self.test_fine_tune)

    def sample_one_task(self, task_source, class_index, K_shot, query_size, test_start_idx=None):

        support_set = []
        query_set = []
        for index in class_index:
            g_list = list(task_source[index])
            if self.args.test_mixup:
                mid = g_list[:K_shot] + list(self.generate_test_graphs[index])
                support_set.append(mid)
            else:
                support_set.append(g_list[:K_shot])

        # during test, sample from all test samples
        append_count = 0
        if task_source == self.test_tasks and test_start_idx != None:
            for i in range(len(class_index)):
                query_set.append(self.total_test_g_list[
                                 min(test_start_idx + i * query_size, len(self.total_test_g_list)):min(
                                     test_start_idx + (i + 1) * query_size, len(self.total_test_g_list))])
                while len(query_set[-1]) < query_size:
                    query_set[-1].append(query_set[0][-1])
                    append_count += 1

        return {'support_set': support_set, 'query_set': query_set, 'append_count': append_count}


class Graph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor #[2, Number of edges]
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.y_a = -1
        self.y_b = -1
        self.lam = 0

        self.max_neighbor = 0


def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')

    if dataset in ['Letter_high', 'ENZYMES', 'Reddit', 'TRIANGLES']:
        g_list = []
        label_dict = {}
        feat_dict = {}

        with open('../datasets/%s/%s.txt' % (dataset, dataset), 'r') as f:
            n_g = int(f.readline().strip())
            for i in range(n_g):
                row = f.readline().strip().split()
                n, l = [int(w) for w in row]
                if not l in label_dict:
                    mapped = len(label_dict)
                    label_dict[l] = mapped
                g = nx.Graph()
                node_tags = []
                node_features = []
                n_edges = 0
                for j in range(n):
                    g.add_node(j)
                    row = f.readline().strip().split()
                    tmp = int(row[1]) + 2
                    if tmp == len(row):
                        # no node attributes
                        row = [int(w) for w in row]
                        attr = None
                    else:
                        row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    if not row[0] in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[row[0]] = mapped
                    node_tags.append(feat_dict[row[0]])

                    if tmp > len(row):
                        node_features.append(attr)

                    n_edges += row[1]
                    for k in range(2, len(row)):
                        g.add_edge(j, row[k])

                if node_features != []:
                    node_features = np.stack(node_features)
                    node_feature_flag = True
                else:
                    node_features = None
                    node_feature_flag = False

                assert len(g) == n

                g_list.append(Graph(g, l, node_tags))

        # add labels and edge_mat
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            # g.label = label_dict[g.label]

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])

            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        if degree_as_tag:
            for g in g_list:
                g.node_tags = list(dict(g.g.degree).values())
                if np.sum(np.array(g.node_tags) == 0): print(g.node_tags)

        # Extracting unique tag labels
        tagset = set([])
        for g in g_list:
            tagset = tagset.union(set(g.node_tags))

        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}

        for g in g_list:
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        print('# classes: %d' % len(label_dict))
        print('# maximum node tag: %d' % len(tagset))
        print("# data: %d" % len(g_list), "\n")

        return g_list, label_dict, tagset

    elif dataset in ['R52', 'COIL']:
        print(dataset)
        node_attribures = pkl.load(open('../datasets/{}/{}_node_attributes.pickle'.format(dataset, dataset), 'rb'))
        train_set = pkl.load(open('../datasets/{}/{}_train_set.pickle'.format(dataset, dataset), 'rb'))
        val_set = pkl.load(open('../datasets/{}/{}_val_set.pickle'.format(dataset, dataset), 'rb'))
        test_set = pkl.load(open('../datasets/{}/{}_test_set.pickle'.format(dataset, dataset), 'rb'))
        #         for sets in [train_set, val_set, test_set]:
        #             class_train=set()
        #             for one in sets['label2graphs'].keys():
        #                 class_train.add(one)
        #             print(class_train)

        g_list = []
        for sets in [train_set, val_set, test_set]:
            graph2nodes = sets["graph2nodes"]
            graph2edges = sets['graph2edges']
            label2graphs = sets['label2graphs']
            for label, graphs in label2graphs.items():
                for graph_id in graphs:
                    g = nx.Graph()
                    node_mapping = {}
                    for node in graph2nodes[graph_id]:
                        node_mapping[node] = len(node_mapping)
                        g.add_node(node_mapping[node])
                    for edge in graph2edges[graph_id]:
                        g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
                    g = Graph(g, label)

                    g.neighbors = [[] for i in range(len(g.g))]
                    for i, j in g.g.edges():
                        g.neighbors[i].append(j)
                        g.neighbors[j].append(i)
                    degree_list = []
                    for i in range(len(g.g)):
                        g.neighbors[i] = g.neighbors[i]
                        degree_list.append(len(g.neighbors[i]))
                    g.max_neighbor = max(degree_list)

                    edges = [list(pair) for pair in g.g.edges()]
                    edges.extend([[i, j] for j, i in edges])

                    deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
                    g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

                    g.node_features = torch.FloatTensor(node_attribures[graph2nodes[graph_id]])
                    if dataset == 'R52':
                        g.node_features = g.node_features.unsqueeze(-1)

                    g_list.append(g)

        print("# data: %d" % len(g_list), "\n")
        return g_list, None, None

    elif dataset == 'ogbg-ppa':
        dataset = GraphPropPredDataset(name=dataset)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        g_list = []
        ### set i as an arbitrary index
        for i in range(len(dataset)):

            graph, label = dataset[i]  # graph: library-agnostic graph object

            nx_graph = nx.Graph()
            for j in range(graph['num_nodes']):
                nx_graph.add_node(j)
            for j in range(graph['edge_index'].shape[1]):
                nx_graph.add_edge(graph['edge_index'][j, 0], graph['edge_index'][j, 1])

            g = Graph(nx_graph, label)
            g.edge_mat = torch.LongTensor(graph['edge_index'])
            g.node_features = torch.FloatTensor(graph['node_feat'])

            g_list.append(g)
            tagset = [i for i in range(37)]

        return g_list, {i: i for i in range(37)}, tagset
