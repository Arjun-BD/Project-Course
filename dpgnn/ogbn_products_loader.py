import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
from ogb.nodeproppred import NodePropPredDataset
from .utils import SparseRowIndexer

def get_data_ogbn_products(dataset_path, privacy_amplify_sampling_rate):
    _split_property = 'split/sales_ranking/'

    train_split_file = os.path.join(dataset_path, _split_property, 'train.csv.gz')
    test_split_file = os.path.join(dataset_path, _split_property, 'test.csv.gz')

    attr_matrix = np.loadtxt(os.path.join(dataset_path, 'raw/node-feat.csv.gz'), delimiter=',')
    labels = np.loadtxt(os.path.join(dataset_path, 'raw/node-label.csv.gz'), delimiter=',')
    edge = np.loadtxt(os.path.join(dataset_path, 'raw/edge.csv.gz'), delimiter=',')

    n, d = attr_matrix.shape
    class_number = len(np.unique(labels))

    train_total_idx = np.loadtxt(train_split_file, delimiter=',', dtype=np.int32)
    test_idx = np.loadtxt(test_split_file, delimiter=',', dtype=np.int32)

    # Use subsamples from all train nodes for actual training (privacy amplification)
    train_total_idx = np.random.permutation(train_total_idx)
    train_idx = train_total_idx[:int(np.ceil(privacy_amplify_sampling_rate * len(train_total_idx)))]

    graph = nx.Graph()
    graph.add_edges_from(edge.astype(int))
    adj_matrix = nx.to_scipy_sparse_matrix(graph)

    # Cut the graph
    graph_train = _get_graph_for_split_with_self_loop(adj_matrix, train_idx)
    graph_test = _get_graph_for_split_with_self_loop(adj_matrix, test_idx)

    # Generate train subgraph
    train_labels = labels[train_idx]
    train_adj_matrix = nx.to_scipy_sparse_matrix(graph_train)
    train_attr_matrix = sp.csr_matrix(attr_matrix[train_idx, :])
    train_index = np.arange(len(train_idx))
    num_edges = graph_train.number_of_edges()

    # Generate test subgraph
    test_labels = labels[test_idx]
    test_adj_matrix = nx.to_scipy_sparse_matrix(graph_test)
    test_attr_matrix = sp.csr_matrix(attr_matrix[test_idx, :])
    if sp.issparse(test_attr_matrix):
        test_attr_matrix = SparseRowIndexer(test_attr_matrix)
    test_index = np.arange(len(test_idx))

    return train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
           test_attr_matrix, test_index, n, class_number, d, num_edges


def _get_graph_for_split_with_self_loop(adj_full, split_set):
    """Returns the induced subgraph for the required split."""
    def edge_generator():
        senders, receivers = adj_full.nonzero()
        for sender, receiver in zip(senders, receivers):
            if sender in split_set and receiver in split_set:
                yield sender, receiver

    def self_loop_generator():
        for idx in split_set:
            yield idx, idx

    graph_split = nx.Graph()
    graph_split.add_nodes_from(split_set)
    graph_split.add_edges_from(edge_generator())
    graph_split.add_edges_from(self_loop_generator())
    return graph_split


def main():
    dataset_path = 'data/ogbn_products'
    privacy_amplify_sampling_rate = 0.1

    # Download and load the dataset
    dataset = NodePropPredDataset(name='ogbn-products')
    graph, labels = dataset[0]

    # Get the graph components
    edge_index = graph['edge_index']
    node_feat = graph['node_feat']
    num_nodes = graph['num_nodes']

    # Save the dataset to files
    os.makedirs(os.path.join(dataset_path, 'raw'), exist_ok=True)
    np.savetxt(os.path.join(dataset_path, 'raw/node-feat.csv.gz'), node_feat, delimiter=',')
    np.savetxt(os.path.join(dataset_path, 'raw/node-label.csv.gz'), labels.squeeze(), delimiter=',')
    np.savetxt(os.path.join(dataset_path, 'raw/edge.csv.gz'), edge_index.T, delimiter=',')

    # Get the split indices
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    test_idx = split_idx['test']

    # Save the split indices
    os.makedirs(os.path.join(dataset_path, 'split/sales_ranking'), exist_ok=True)
    np.savetxt(os.path.join(dataset_path, 'split/sales_ranking/train.csv.gz'), train_idx, delimiter=',')
    np.savetxt(os.path.join(dataset_path, 'split/sales_ranking/test.csv.gz'), test_idx, delimiter=',')

    # Load the dataset using the provided function
    train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
    test_attr_matrix, test_index, n, class_number, d, num_edges = get_data_ogbn_products(dataset_path, privacy_amplify_sampling_rate)

    print("Dataset loaded and processed successfully.")
    print(f"Number of nodes: {n}")
    print(f"Number of attributes: {d}")
    print(f"Number of classes: {class_number}")
    print(f"Number of edges: {num_edges}")

if __name__ == "__main__":
    main()
