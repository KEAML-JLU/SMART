import torch
import copy
import random
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F


def main():
    pass


def aug_drop_node(batch_graph):
    for i, graph in enumerate(batch_graph):
        node_num, _ = graph.node_features.size()
        _, edge_num = graph.edge_mat.size()
        drop_num = int(node_num / 10)

        idx_drop = np.random.choice(node_num, drop_num, replace=False)
        # idx_nondrop = [n for n in range(node_num) if not n in idx_drop]

        edge_index = graph.edge_mat.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj[idx_drop, :] = 0
        adj[:, idx_drop] = 0
        edge_index = adj.nonzero().t()
        graph.edge_mat = edge_index
        # check(graph)

    return batch_graph


def aug_fea_mask(batch_graph):
    for i, graph in enumerate(batch_graph):
        node_num, feat_dim = graph.node_features.size()
        mask_num = int(node_num / 10)

        idx_mask = np.random.choice(node_num, mask_num, replace=False)
        graph.node_features[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)),
                                                     dtype=torch.float32)

    return batch_graph


def aug_fea_drop(batch_graph):
    for i, graph in enumerate(batch_graph):
        drop_mask = torch.empty((graph.node_features.size(1),), dtype=torch.float32).uniform_(0, 1) < 0.1
        graph.node_features = graph.node_features.clone()
        graph.node_features[:, drop_mask] = 0
    return batch_graph


def aug_fea_dropout(batch_graph):
    for i, graph in enumerate(batch_graph):
        graph.node_features = F.dropout(graph.node_features, p=0.1)
    return batch_graph


def check(graph):
    node_num, _ = graph.node_features.size()
    edge_idx = graph.edge_mat.numpy()
    _, edge_num = edge_idx.shape
    idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

    node_num_aug = len(idx_not_missing)
    graph.node_features = graph.node_features[idx_not_missing]

    # data_aug.batch = data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
    edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                not edge_idx[0, n] == edge_idx[1, n]]
    graph.edge_mat = torch.LongTensor(edge_idx).transpose_(0, 1)


if __name__ == "__main__":
    main()
