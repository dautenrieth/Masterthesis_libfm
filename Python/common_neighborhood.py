import torch
from tqdm import tqdm
from torch_geometric.data import DataLoader
import scipy.sparse as ssp
import numpy as np


def CN(A, edge_index):
    src, dst = edge_index[0], edge_index[1]
    cur_score = int((np.sum(A[src].multiply(A[dst]), 1)).flatten())
    return cur_score


def CN_array(A, edge_index, binary=False):
    src, dst = edge_index[0], edge_index[1]
    Neighboors = A[src].multiply(A[dst])
    # Use Matrix. Every instance that is not 0 will be 1
    if binary:
        Neighboors[Neighboors != 0] = 1
    return Neighboors


def sparse_matrix(data):
    num_nodes = data.num_nodes
    edge_weight = data.edge_weight

    if edge_weight is None:
        edge_weight = np.ones(data.edge_index.shape[1])
    else:
        edge_weight = edge_weight.view(-1)

    A = ssp.csr_matrix(
        (edge_weight, (data.edge_index[0], data.edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    return A


# def sparse_matrix(data):
#     print("Constructing graph.")
#     train_edges_raw = np.array(data.edge_index)
#     train_edges_reverse = np.array(
#         [train_edges_raw[:, 1], train_edges_raw[:, 0]]
#     ).transpose()
#     train_edges = np.concatenate([train_edges_raw, train_edges_reverse], axis=0)
#     edge_weight = torch.ones(train_edges.shape[0], dtype=int)
#     A = ssp.csr_matrix(
#         (edge_weight, (train_edges[:, 0], train_edges[:, 1])),
#         shape=(data.num_nodes, data.num_nodes),
#     )
#     return A
