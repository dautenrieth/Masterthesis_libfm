from pathlib import Path
import json
import time, torch
from tqdm import tqdm
import scipy.sparse as ssp
import networkx as nx
from torch.utils.data import DataLoader
from logger import logging_setup

logger = logging_setup(__name__)
import numpy as np

from common_neighborhood import CN, CN_array

## TODO:
# - Sanitiy check for loaded neighborhood data
# - Check if nodes are in a row
# - Check if nodes start at 0 or 1


def weights_to_line(weights, index):
    # Add Weight at the beginning
    if weights is None:
        line = f"{0} "
    else:
        line = f"{weights[index]} "

    return line


def embeddings_to_line(line, edge, edge_emb, FeatureCounter, first_run, groupinfo=None):
    # edge = [node1, node2]
    # edge to node
    node1, node2 = edge[0], edge[1]

    # Log features
    if first_run:
        # -1 necessary because 0 is included because this is the first feature
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+len(edge_emb[node1])-1}: Edge Embeddings Node 1"
        )
        logger.info(
            f"Features {FeatureCounter+len(edge_emb[node1])-1}-"
            f"{FeatureCounter+len(edge_emb[node1])+len(edge_emb[node2])-1}: Edge Embeddings Node 2"
        )

    # List node embeddings
    # Embeddings node 1
    for val in edge_emb[node1]:
        if val != 0.0:
            line += f"{FeatureCounter}:{val} "
        FeatureCounter += 1

    # Embeddings node 2
    for val in edge_emb[node2]:
        if val != 0.0:
            line += f"{FeatureCounter}:{val} "
        FeatureCounter += 1

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(
                groupinfo, len(edge_emb[node1]) + len(edge_emb[node2])
            )

    return line, FeatureCounter, groupinfo


def ids_to_line(line, edge, FeatureCounter, number_of_nodes, first_run, groupinfo=None):
    if first_run:
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Node IDs"
        )

    # Add Node IDs
    node1, node2 = edge[0], edge[1]
    if node1 < node2:
        line += f"{FeatureCounter+node1}:1 "
        line += f"{FeatureCounter+node2}:1 "
    else:
        line += f"{FeatureCounter+node2}:1 "
        line += f"{FeatureCounter+node1}:1 "

    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo


def neighborhood_to_line(
    line,
    node,
    FeatureCounter,
    number_of_nodes,
    first_run,
    groupinfo=None,
    loaded_data=None,
):
    node = str(int(node))
    # Log features
    if first_run:
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Neighborhood of edge nodes"
        )

    # Find out if nodes are neighbors of edge nodes and add to line
    for neighbor_node in loaded_data[node]:
        line += f"{FeatureCounter+int(neighbor_node)}:1 "
    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo


def recent_neighborhood_to_line(
    line,
    node,
    FeatureCounter,
    number_of_nodes,
    first_run,
    groupinfo=None,
    loaded_data=None,
):
    node = str(int(node))
    # Log features
    if first_run:
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Neighborhood of edge nodes"
        )

    # Find out if nodes are neighbors of edge nodes and add to line
    for neighbor_node in loaded_data[node]:
        line += f"{FeatureCounter+int(neighbor_node)}:1 "
    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo


def neighborhood_data_loader(edge_set, number_of_nodes, config):
    start_time = time.time()
    # Set the path to the folder containing the JSON file
    file_path = Path(
        f"{config['FOLDERNAMES']['neighborhood_folder']}/{config['FILENAMES']['neighborhood_file']}"
    )

    # Check if the file exists in the folder
    if file_path.is_file():
        # Open the file and load the JSON data into a dictionary
        with file_path.open(mode="r") as f:
            data = json.load(f)
        logger.info("Neighborhood data loaded from file.")
    else:
        data = {}
        logger.info("Neighborhood data not found. Creating new data.")
        set_tensor = set(map(tuple, edge_set.numpy()))
        # Create data
        for node in range(0, number_of_nodes):
            data[node] = {}
            for i in range(0, number_of_nodes):  # i = potential neighbor node
                if ((node, i) in set_tensor or (i, node) in set_tensor) and node != i:
                    data[node][i] = 1
        with open(file_path, "w") as json_file:
            json.dump(data, json_file)
        logger.info(
            f"Neighborhood data creation execution time: {time.time() - start_time:.2f} seconds for {number_of_nodes} nodes"
        )
    return data


def add_groupinfo(groupinfo, number_elements):
    groupinfo[1].extend([str(groupinfo[0]) for i in range(number_elements)])
    groupinfo[0] += 1

    return groupinfo


def common_neighborhood_to_line(
    line, FeatureCounter, first_run, edges, s_matrix, groupinfo=None
):
    score = CN(s_matrix, edges)

    line += f"{FeatureCounter}:{score} "
    FeatureCounter += 1

    # Add groupinfos to groupinfo list
    if first_run:
        logger.info(f"Features {FeatureCounter}: Common Neighbors")
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, 1)

    return line, FeatureCounter, groupinfo


def neighborhood_binary_to_line(
    line,
    FeatureCounter,
    number_of_nodes,
    first_run,
    edge,
    s_matrix,
    groupinfo,
    binary=True,
):
    CN_b_array = CN_array(s_matrix, edge, binary=binary)
    for index in range(len(CN_b_array.data)):
        line += f"{FeatureCounter+CN_b_array.indices[index]}:{CN_b_array.data[index]} "

    FeatureCounter += number_of_nodes

    # Add groupinfos to groupinfo list
    if first_run:
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+number_of_nodes}: Common Neighbors Binary"
        )
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, number_of_nodes)

    return line, FeatureCounter, groupinfo


def SumAdamic_to_line(line, FeatureCounter, first_run, edge, nx_graph, groupinfo):
    u, v = edge[0] - 1, edge[1] - 1

    # Check if u and v are in the graph
    if not nx_graph.has_node(u) or not nx_graph.has_node(v):
        similarity = 0
    else:
        # Check if this edge is part of a triangle
        common_neighbors = list(nx.common_neighbors(nx_graph, u, v))
        if len(common_neighbors) == 0:
            similarity = 0
        else:
            # Compute the Adamic/Adar similarity for this edge
            similarity = sum(
                1 / np.log(nx.degree(nx_graph, w)) for w in common_neighbors
            )

    line += f"{FeatureCounter}:{similarity} "

    FeatureCounter += 1

    if first_run:
        logger.info(f"Features {FeatureCounter}-{FeatureCounter+1}: Adamic Adar Sum")
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, 1)

    return line, FeatureCounter, groupinfo


def resource_allocation_to_line(
    line, FeatureCounter, first_run, edge, s_matrix, groupinfo
):
    """
    cite: [Predicting missing links via local information](https://arxiv.org/pdf/0901.0553.pdf)
    Code from: https://github.com/CUAI/Edge-Proposal-Sets
    :param adj_matrix: Compressed Sparse Row matrix
    :param link_list: torch tensor list of links, shape[m, 2]
    :return: RA similarity for each link
    """

    w = 1 / s_matrix.sum(axis=0)
    w[np.isinf(w)] = 0
    D = s_matrix.multiply(w).tocsr()  # e[i,j] / log(d_j)

    src, dst = edge
    score = np.array(np.sum(s_matrix[src].multiply(D[dst]), 1)).flatten()

    line += f"{FeatureCounter}:{score} "
    FeatureCounter += 1

    if first_run:
        logger.info(
            f"Features {FeatureCounter}-{FeatureCounter+1}: Ressource Allocation"
        )
        if groupinfo is not None:
            groupinfo = add_groupinfo(groupinfo, 1)

    return line, FeatureCounter, groupinfo
