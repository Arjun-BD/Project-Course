import os
import logging
import numpy as np
import time
import math
from absl import app
from absl import flags
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from dpgnn import utils
from dpgnn import ppr_utils as ppr
from dpgnn.model import DPGNN, train as train_solo
from dpgnn.model_enc import DPGNN as DPGNN_enc, train as train_w_enc, train_encoder
from tqdm import tqdm
import pandas as pd
from pynverse import inversefunc
import pickle as pkl
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_v2_behavior()

# Flags configuration (keep all existing flags)
flags.DEFINE_string('data_file', 'data/cora_ml', 'Path to the .npz data file')
flags.DEFINE_float('lr', 5e-3, 'learning rate')
flags.DEFINE_float('alpha', 0.25, 'PPR teleport probability')
flags.DEFINE_float('rho', 1e-4, 'ISTA hyparameter rho')
flags.DEFINE_float('eps', 1e-4, 'ISTA hyparameter eps')
flags.DEFINE_integer('topk', 4, 'Number of PPR neighbors for each node')
flags.DEFINE_bool('dp_ppr', False, 'Enable DP-PPR or not')
flags.DEFINE_bool('EM', False, 'Enable EM or not')
flags.DEFINE_bool('dp_sgd', False, 'Enable DP-SGD or not')
flags.DEFINE_integer('ppr_num', 70, 'Number of nodes within sampled that are calculated their ppr vectors')
flags.DEFINE_float('privacy_amplify_sampling_rate', 0.09, 'Privacy amplification sampling rate')
flags.DEFINE_float('delta', 1e-4, 'DP budget delta for private ppr')

# Additional flags for experiment
flags.DEFINE_integer('num_iterations', 10, 'Number of iterations for each experiment')

# For output perturbation of dp-ppr using Gaussian Mechanism
flags.DEFINE_float('sigma_ista', 0.0067, 'Gaussian noise standard deviation for DP-ISTA')
flags.DEFINE_float('clip_bound_ista', 0.01, 'Gradient clip bound for DP-ISTA')
flags.DEFINE_float('EM_eps', 1.3, 'Privacy budget epsilon for EM')
flags.DEFINE_float('report_val_eps', -1.0, 'Privacy budget epsilon for reporting topk noise values for each node')
flags.DEFINE_float('em_sensitivity', 0.001, 'Global sensitivity (each ppr value will be clipped into [0, em_sensitivity]) for EM')

# For DP-GNN model training
flags.DEFINE_float('sigma_sgd', 0.95, 'Gaussian noise standard deviation for DP-SGD')
flags.DEFINE_float('delta_sgd', 1e-3, 'Privacy budget delta for DP-SGD')
flags.DEFINE_float('clip_bound_sgd', 1, 'Gradient clip bound for DP-SGD')
flags.DEFINE_integer('microbatches', 60, 'Number of microbatches (must evenly divide batch_size for training)')

# For Encoder
flags.DEFINE_bool("use_encoder", False, 'Use Encoder for node feature embeddings')
flags.DEFINE_integer('nEncLayers', 0, 'Number of encoder layers')
flags.DEFINE_integer('nEncOut', 32, 'Size of encoded dimensions')
flags.DEFINE_float('EncDropout', 0.1, 'Encoder dropout rate')
flags.DEFINE_integer('encoder_epochs', 10, 'Encoder training epochs')
flags.DEFINE_bool('Enc_dp_sgd', True, 'Enable DP-SGD or not')
flags.DEFINE_bool('use_enc_embed', False, 'Use Encoded embedding by concatenating neural network layers')

flags.DEFINE_string('ppr_normalization', 'row', 'Adjacency matrix normalization for weighting neighbors')
flags.DEFINE_integer('hidden_size', 32, 'Size of the MLP hidden layer')
flags.DEFINE_integer('nlayers', 2, 'Number of MLP layers')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay used for training the MLP')
flags.DEFINE_float('dropout', 0.1, 'Dropout used for training')
flags.DEFINE_integer('max_epochs', 200, 'Maximum number of epochs (exact number if no early stopping)')
flags.DEFINE_integer('batch_size', 60, 'Batch size for training')
flags.DEFINE_integer('nprop_inference', 2, 'Number of propagation steps during inference')

FLAGS = flags.FLAGS

def compute_feature_centers(labels, features, indices):
    """
    Compute feature centers for nodes with the same label.
    """
    labels = np.array(labels)
    indices = np.array(indices)
    
    node_labels = labels[indices]
    node_features = features[indices]
    
    unique_labels = np.unique(node_labels)
    
    centers = []
    for label in unique_labels:
        rows = node_features[node_labels == label]
        center = rows.mean(axis=0)
        centers.append(center)
    
    return csr_matrix(np.vstack(centers))

def count_elements(arr: np.ndarray) -> dict:
    """
    Count occurrences of unique elements in an array.
    """
    unique_elements, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique_elements, counts))

def clus(centers, vec, k=2, mode='nearest'):
    """
    Cluster selection function.
    """
    if mode not in {'nearest', 'farthest', 'random'}:
        raise ValueError("Mode must be 'nearest', 'farthest', or 'random'.")
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((centers - vec) ** 2, axis=1))
    
    if mode == 'nearest':
        # Get indices of k smallest distances
        indices = np.argsort(distances)[:k]
    elif mode == 'farthest':
        # Get indices of k largest distances
        indices = np.argsort(-distances)[:k]
    elif mode == 'random':
        # Select k random indices
        indices = np.random.choice(len(centers), size=k, replace=False)
    
    return indices.tolist()

def run_experiment(train_labels, train_adj_matrix, train_attr_matrix, train_index, 
                   test_labels, test_adj_matrix, test_attr_matrix, test_index, 
                   centers, d, nc, k, mode, use_enc=False):
    """
    Run a single experiment with specified parameters.
    """
    tf.reset_default_graph()
    tf.set_random_seed(np.random.randint(1000))
    
    # Prepare topk matrix
    my_topk = [[0.0 for i in range(len(train_adj_matrix.toarray()))] for i in range(len(train_adj_matrix.toarray()))]
    
    element_counts = count_elements(train_labels)

    for i in range(len(train_adj_matrix.toarray())):
        cen = clus(centers.toarray(), train_attr_matrix.toarray()[i], k=k, mode=mode)
        for j in range(len(train_labels)):
            if train_labels[j] in cen:
                my_topk[i][j] = 1/(element_counts[train_labels[j]])
            
    my_topk = sp.csr_matrix(my_topk)

    # Normalize l1 norm of each column of topk
    my_topk_dense = my_topk.toarray()
    for col in range(len(my_topk_dense[0, :])):
        if np.linalg.norm(my_topk_dense[:, col], ord=1) != 0:
            my_topk_dense[:, col] *= (1.0 / np.linalg.norm(my_topk_dense[:, col], ord=1))
    my_topk = sp.csr_matrix(my_topk_dense)
    
    # Model setup
    if use_enc:
        model = DPGNN_enc(d, nc, FLAGS.hidden_size, FLAGS.nlayers, FLAGS.nEncLayers, FLAGS.nEncOut, 
                          FLAGS.lr, FLAGS.weight_decay, FLAGS.dropout, FLAGS.EncDropout,
                          FLAGS.clip_bound_sgd, FLAGS.sigma_sgd, FLAGS.microbatches, 
                          dp_sgd=FLAGS.dp_sgd, enc_dp_sgd=FLAGS.Enc_dp_sgd,
                          sparse_features=type(train_attr_matrix) is not np.ndarray)
        train_func = train_w_enc
    else:
        model = DPGNN(d, nc, FLAGS.hidden_size, FLAGS.nlayers, FLAGS.lr, FLAGS.weight_decay, 
                      FLAGS.dropout, FLAGS.clip_bound_sgd, FLAGS.sigma_sgd, FLAGS.microbatches, 
                      dp_sgd=FLAGS.dp_sgd, sparse_features=type(train_attr_matrix) is not np.ndarray)
        train_func = train_solo

    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()

        if use_enc:
            for epoch in range(FLAGS.encoder_epochs):
                random_index = np.random.permutation(len(train_labels))
                train_index_shuffled = train_index[random_index]

                train_encoder(
                    sess=sess, model=model, attr_matrix=train_attr_matrix,
                    train_idx=train_index_shuffled, topk_train=my_topk, labels=train_labels,
                    epoch=epoch, batch_size=FLAGS.batch_size)

        for epoch in range(FLAGS.max_epochs):
            random_index = np.random.permutation(len(train_labels))
            train_index_shuffled = train_index[random_index]

            train_func(
                sess=sess, model=model, attr_matrix=train_attr_matrix,
                train_idx=train_index_shuffled, topk_train=my_topk, labels=train_labels,
                epoch=epoch, batch_size=FLAGS.batch_size)

        # Inference
        predictions = model.predict(
            sess=sess, adj_matrix=test_adj_matrix, attr_matrix=test_attr_matrix, 
            alpha=FLAGS.alpha, nprop=FLAGS.nprop_inference, 
            ppr_normalization=FLAGS.ppr_normalization)
        
        test_acc = accuracy_score(test_labels, predictions)
        
        sess.close()
        return test_acc

def plot_accuracy_comparison(train_labels, train_adj_matrix, train_attr_matrix, train_index, 
                             test_labels, test_adj_matrix, test_attr_matrix, test_index, 
                             centers, d, nc):
    """
    Plot accuracy comparison for different k values and clustering modes.
    """
    # k_values = [2, 3, 4]
    # modes = ['nearest', 'farthest', 'random']
    k_values = [2,3,4]
    modes = ['nearest','farthest', 'random']
    # Store results
    results = {mode: {k: [] for k in k_values} for mode in modes}
    
    # Total number of experiments for progress bar
    total_experiments = len(modes) * len(k_values) * FLAGS.num_iterations
    
    # Progress bar
    with tqdm(total=total_experiments, desc="Running Experiments", unit="experiment") as pbar:
        # Run experiments
        for mode in modes:
            for k in k_values:
                print(f"\nRunning experiments: mode={mode}, k={k}")
                mode_k_accuracies = []
                for _ in range(FLAGS.num_iterations):
                    acc = run_experiment(
                        train_labels, train_adj_matrix, train_attr_matrix, train_index, 
                        test_labels, test_adj_matrix, test_attr_matrix, test_index, 
                        centers, d, nc, k, mode
                    )
                    mode_k_accuracies.append(acc)
                    pbar.update(1)  # Update progress bar
                results[mode][k] = mode_k_accuracies
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Color and marker mappings
    mode_colors = {
        'nearest': 'blue', 
        'farthest': 'red', 
        'random': 'green'
    }
    
    mode_markers = {
        'nearest': 'o', 
        'farthest': 's', 
        'random': '^'
    }
    
    # Plot results
    for mode in modes:
        x_values = list(k_values)
        y_values = [np.mean(results[mode][k]) for k in k_values]
        y_errors = [np.std(results[mode][k]) for k in k_values]
        
        plt.errorbar(x_values, y_values, yerr=y_errors, 
                     label=f'{mode} mode', 
                     color=mode_colors[mode], 
                     marker=mode_markers[mode],
                     capsize=5, 
                     linestyle='-', 
                     markersize=8)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy for Different Clustering Modes and k Values', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main(unused_argv):
    ''' Load Data '''
    start = time.time()

    FLAGS.microbatches = FLAGS.batch_size

    train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
    test_attr_matrix, test_index, num_nodes, num_class, num_attr, num_edges = utils.get_data(
        FLAGS.data_file, privacy_amplify_sampling_rate=FLAGS.privacy_amplify_sampling_rate)
    
    # Compute feature centers
    centers = compute_feature_centers(train_labels, train_attr_matrix, train_index)
    
    d = num_attr
    nc = num_class
    time_loading = time.time() - start
    print(f"Runtime: {time_loading:.2f}s")
    print("Finish Load Data.\n")
    
    # Plot accuracy comparison
    plot_accuracy_comparison(
        train_labels, train_adj_matrix, train_attr_matrix, train_index, 
        test_labels, test_adj_matrix, test_attr_matrix, test_index, 
        centers, d, nc
    )

if __name__ == '__main__':
    app.run(main)