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

# Set up logging

import torch
import sys
sys.path.append('./GAE/R-GMM-VGAE/')  # Add the path to the folder containing `model_cora.py`
from model_cora import ReGMM_VGAE, clustering_metrics
from datasets import format_data
from preprocessing import load_data, sparse_to_tuple, preprocess_graph

def predict_node_labels(model_path, dataset="Cora", data_path="./data/npz", save_predictions=True,clus = 7):
    # Network parameters
    num_neurons = 32
    embedding_size = 16
    nClusters = clus  # Specific to Cora dataset
    
    # Load and preprocess data
    feas = format_data(dataset.lower(), data_path)
    num_features = feas['features'].size(1)
    adj, features, true_labels = load_data(dataset.lower(), data_path)
    
    # Process adjacency matrix
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm = preprocess_graph(adj)
    
    # Convert features to sparse tensor format
    features = sparse_to_tuple(features.tocoo())
    features = torch.sparse.FloatTensor(
        torch.LongTensor(features[0].T), 
        torch.FloatTensor(features[1]), 
        torch.Size(features[2])
    )
    
    # Convert adjacency matrix to tensor
    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T), 
        torch.FloatTensor(adj_norm[1]), 
        torch.Size(adj_norm[2])
    )
    
    # Initialize model
    model = ReGMM_VGAE(
        num_neurons=num_neurons,
        num_features=num_features,
        embedding_size=embedding_size,
        nClusters=nClusters
    )
    
    # Load saved model state
    model.load_state_dict(torch.load(model_path))
    # model.eval()
    
    # Get embeddings and predictions
    with torch.no_grad():
        _, _, embeddings = model.encode(features, adj_norm)
        predictions = model.predict(embeddings)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Node_ID': range(len(predictions)),
        'Predicted_Cluster': predictions,
        'True_Label': true_labels
    })
    
    # Save predictions to CSV
    if save_predictions:
        output_path = f"node_predictions_{dataset.lower()}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results_df, embeddings,true_labels

MODEL_PATH = "./saved_models/ReGMM_VGAE_model.pth"
    
# Get predictions for all nodes
predictions_df, embeddings,true_labels = predict_node_labels(
    model_path=MODEL_PATH,
    dataset="Cora",
    data_path="GAE/R-GMM-VGAE/data/Cora"
)
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
accuracy = accuracy_score(predictions_df["True_Label"], predictions_df["Predicted_Cluster"])

print("Accuracy:", accuracy)
# def calculate_accuracy(predictions_df):
#     # Extract true labels and predicted clusters
#     true_labels = predictions_df["True_Label"]
#     predicted_clusters = predictions_df["Predicted_Cluster"]

#     # Get unique true labels and predicted clusters
#     unique_true_labels = np.unique(true_labels)
#     unique_predicted_clusters = np.unique(predicted_clusters)

#     # Create cost matrix
#     cost_matrix = np.zeros((len(unique_true_labels), len(unique_predicted_clusters)))
#     for i, true_label in enumerate(unique_true_labels):
#         for j, cluster in enumerate(unique_predicted_clusters):
#             # Negative count for correct matches
#             cost_matrix[i, j] = -np.sum((true_labels == true_label) & (predicted_clusters == cluster))

#     # Solve the assignment problem
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)

#     # Map clusters to true labels
#     cluster_to_label_mapping = {unique_predicted_clusters[col]: unique_true_labels[row]
#                                  for row, col in zip(row_ind, col_ind)}

#     # Map predicted clusters to corresponding true labels
#     mapped_predictions = predicted_clusters.map(cluster_to_label_mapping)

#     # Calculate accuracy
#     accuracy = accuracy_score(true_labels, mapped_predictions)
#     return accuracy, cluster_to_label_mapping

# # Calculate accuracy and mapping
# accuracy, cluster_mapping = calculate_accuracy(predictions_df)

# print("Accuracy:", accuracy)
# print("Cluster-to-Label Mapping:", cluster_mapping)

# Print first few predictions
print("\nFirst 10 node predictions:")
print(predictions_df.head(10))

# Print cluster distribution
print("\nCluster distribution:")
print(predictions_df['Predicted_Cluster'].value_counts().sort_index())

logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel('INFO')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

''' Configuration '''
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

# For output perturbation of dp-ppr using Gaussian Mechanism
flags.DEFINE_float('sigma_ista', 0.0067, 'Gaussian noise standard deviation for DP-ISTA')
flags.DEFINE_float('clip_bound_ista', 0.01, 'Gradient clip bound for DP-ISTA')
flags.DEFINE_float('EM_eps', 1.3, 'Privacy budget epsilon for EM')
flags.DEFINE_float('report_val_eps', -1.0, 'Privacy budget epsilon for reporting topk noise values for each node')
flags.DEFINE_float('em_sensitivity', 0.001, 'Global sensitivity (each ppr value will be clipped into '
                                                    '[0, em_sensitivity]) for EM')
# For DP-GNN model training
flags.DEFINE_float('sigma_sgd', 0.95, 'Gaussian noise standard deviation for DP-SGD')
flags.DEFINE_float('delta_sgd', 1e-3, 'Privacy budget delta for DP-SGD')
flags.DEFINE_float('clip_bound_sgd', 1, 'Gradient clip bound for DP-SGD')
flags.DEFINE_integer('microbatches', 60, 'Number of microbatches (must evenly divide batch_size for training)')

# For Encoder
flags.DEFINE_bool("use_encoder", False,'Use Encoder for node feature embeddings')
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


def main(unused_argv):
    ''' Load Data '''
    start = time.time()

    FLAGS.microbatches = FLAGS.batch_size

    train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
    test_attr_matrix, test_index, num_nodes, num_class, num_attr, num_edges = utils.get_data(FLAGS.data_file,
                                                     privacy_amplify_sampling_rate=FLAGS.privacy_amplify_sampling_rate)
    def compute_feature_centers(labels, features, indices):
        """
        Compute feature centers for nodes with the same label.

        Args:
            labels (ndarray): Array of labels for all nodes.
            features (csr_matrix): Sparse matrix of features (shape: num_nodes x num_features).
            indices (ndarray): Indices of nodes to consider.

        Returns:
            csr_matrix: A sparse matrix where each row is the center for a label.
        """
        # Ensure inputs are NumPy arrays
        labels = np.array(labels)
        indices = np.array(indices)
        
        # Extract the labels and features for the specified indices
        node_labels = labels[indices]
        node_features = features[indices]
        
        # Find unique labels
        unique_labels = np.unique(node_labels)
        
        # Compute centers
        centers = []
        for label in unique_labels:
            # Get rows corresponding to the current label
            rows = node_features[node_labels == label]
            # Compute the mean (center) for this label
            center = rows.mean(axis=0)
            centers.append(center)
        
        # Stack centers into a matrix
        return csr_matrix(np.vstack(centers))
    

    centers = compute_feature_centers(train_labels, train_attr_matrix, train_index)
    
    print(centers.toarray())  # Convert sparse matrix to dense for display


    d = num_attr
    nc = num_class
    time_loading = time.time() - start
    print(f"Runtime: {time_loading:.2f}s")
    print("Finish Load Data.\n")
    epsilon = 0
    eps_sgd = 0
    ppr_num = FLAGS.ppr_num

    use_enc = FLAGS.use_encoder

    """ Calculate Privacy Budget """
    ' Calculate Privacy Budget epsilon for DP-PPR '
    if FLAGS.dp_ppr == True:
        eps_ista = np.sqrt(2 * np.log(1.25 / FLAGS.delta)) * FLAGS.clip_bound_ista / FLAGS.sigma_ista
        delta = 2 * FLAGS.delta * ppr_num
        composition = (lambda x: x / (2 * np.sqrt(ppr_num * np.log(np.e + x / delta))))
        epsilon = inversefunc(composition, y_values=eps_ista)
        epsilon *= FLAGS.privacy_amplify_sampling_rate
        delta *= FLAGS.privacy_amplify_sampling_rate
        privacy_record = f'''DP budget for DP-PPR using Gaussian Mechanism: epsilon={epsilon:.4f}, delat={delta}'''
        print(privacy_record)
    elif FLAGS.EM == True:
        eps_prime = min(FLAGS.topk * FLAGS.EM_eps,
                        FLAGS.topk * FLAGS.EM_eps * (math.exp(FLAGS.EM_eps) - 1) / (math.exp(FLAGS.EM_eps) + 1) + FLAGS.EM_eps * np.sqrt(
                            2 * FLAGS.topk * np.log(1.0 / FLAGS.delta)))
        if FLAGS.report_val_eps < 0:
            FLAGS.report_val_eps = float(eps_prime)
        eps_prime += FLAGS.report_val_eps
        delta = 2 * ppr_num * FLAGS.delta
        composition = (lambda x: x / (2 * np.sqrt(ppr_num * np.log(np.e + x / delta))))
        epsilon = inversefunc(composition, y_values=eps_prime)
        epsilon *= FLAGS.privacy_amplify_sampling_rate
        delta *= FLAGS.privacy_amplify_sampling_rate
        privacy_record = f'''DP budget for DP-PPR using EM: epsilon={epsilon:.4f}, delat={delta}'''
        print(privacy_record)
    else:
        privacy_record = "No DP for calculating PPR"
        print(privacy_record)

    if FLAGS.dp_sgd:
        ' Calculate Privacy Budget epsilon for DP-SGD '
        num_steps = FLAGS.max_epochs * np.ceil(len(train_index) / FLAGS.batch_size)
        sampling_rate = float(FLAGS.batch_size) / len(train_index)
        delta_sgd = 0.001 / FLAGS.privacy_amplify_sampling_rate
        noise_multiplier = FLAGS.sigma_sgd / FLAGS.clip_bound_sgd
        eps_sgd = utils.compute_epsilon(num_steps, noise_multiplier, delta_sgd, sampling_rate)
        eps_sgd = eps_sgd * FLAGS.privacy_amplify_sampling_rate
        delta = delta_sgd * FLAGS.privacy_amplify_sampling_rate
        privacy_record_sgd = f'''DP budget for DP-SGD: epsilon={eps_sgd:.4f}, delta={delta}'''
        print(privacy_record_sgd)
    else:
        privacy_record_sgd = "No DP when doing SGD"
        print(privacy_record_sgd)

    if use_enc:
        if FLAGS.Enc_dp_sgd:
            ' Calculate Privacy Budget epsilon for DP-SGD '
            num_steps = FLAGS.max_epochs * np.ceil(len(train_index) / FLAGS.batch_size)
            sampling_rate = float(FLAGS.batch_size) / len(train_index)
            delta_sgd = 0.001 / FLAGS.privacy_amplify_sampling_rate
            noise_multiplier = FLAGS.sigma_sgd / FLAGS.clip_bound_sgd
            eps_sgd = utils.compute_epsilon(num_steps, noise_multiplier, delta_sgd, sampling_rate)
            eps_sgd = eps_sgd * FLAGS.privacy_amplify_sampling_rate
            delta = delta_sgd * FLAGS.privacy_amplify_sampling_rate
            privacy_record_sgd = f'''DP budget for DP-SGD on Encoder: epsilon={eps_sgd:.4f}, delat={delta}'''
            print(privacy_record_sgd)
        else:
            privacy_record_sgd = "No DP-SGD on Encoder"
            print(privacy_record_sgd)


    ''' Preprocessing: Calculate PPR scores '''
    # start = time.time()
    # topk_train = ppr.topk_ppr_matrix_ista(train_adj_matrix, FLAGS.alpha, FLAGS.eps, FLAGS.rho, train_index[:ppr_num],
    #                                         FLAGS.topk, FLAGS.sigma_ista, FLAGS.clip_bound_ista, FLAGS.dp_ppr,
    #                                         FLAGS.em_sensitivity, FLAGS.report_val_eps, FLAGS.EM, FLAGS.EM_eps)
    # print(topk_train.nnz,len(topk_train.toarray()))
    # if ppr_num < len(train_index):
    #     topk_train_I = np.identity(len(train_index))
    #     topk_train_I = topk_train_I[ppr_num:]
    #     topk_train_dense = topk_train.toarray()
    #     topk_train_dense_full = np.concatenate((topk_train_dense, topk_train_I), axis=0)
    #     topk_train = sp.csr_matrix(topk_train_dense_full)

    # time_preprocessing = time.time() - start
    # print(f"Calculate Train PPR Matrix Runtime: {time_preprocessing:.2f}s")

    my_topk  = [[0.0 for i in range(len(train_adj_matrix.toarray()))] for i in range(len(train_adj_matrix.toarray()))]
    
    def clus(centers, vec, k= 2, mode = 'nearest'):
        
        if mode not in {'nearest', 'farthest', 'random'}:
            raise ValueError("Mode must be 'nearest', 'farthest', or 'random'.")
        
        # Normalize centers and vec for cosine similarity
        norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
       
        norm_vec = vec / np.linalg.norm(vec)
        
        # Calculate cosine similarity
        similarities = norm_centers @ norm_vec

        if mode == 'nearest':
            # Get indices of k highest similarities
            indices = np.argsort(-similarities)[:k]
        elif mode == 'farthest':
            # Get indices of k lowest similarities
            indices = np.argsort(similarities)[:k]
        elif mode == 'random':
            # Select k random indices
            indices = np.random.choice(len(centers), size=k, replace=False)
        
        return indices.tolist()
    
    def count_elements(arr: np.ndarray) -> dict:
        unique_elements, counts = np.unique(arr, return_counts=True)
        return dict(zip(unique_elements, counts))
    
    element_counts = count_elements(train_labels)

    for i in range(len(train_adj_matrix.toarray())):
        cen = clus(centers.toarray(),train_attr_matrix.toarray()[i])
        for j in range(len(train_labels)):
            if(train_labels[j] in cen):
                my_topk[i][j] = 1/element_counts[train_labels[j]]
            
    my_topk = sp.csr_matrix(my_topk)

    # normalize l1 norm of each column of topk_train'''
    my_topk_dense = my_topk.toarray()
    for col in range(len(my_topk_dense[0, :])):
        if np.linalg.norm(my_topk_dense[:, col], ord=1) != 0:
            my_topk_dense[:, col] *= (1.0 / np.linalg.norm(my_topk_dense[:, col], ord=1))
    my_topk = sp.csr_matrix(my_topk_dense)

    # normalize l1 norm of each column of topk_train'''
    # topk_train_dense = topk_train.toarray()
    # for col in range(len(topk_train_dense[0, :])):
    #     if np.linalg.norm(topk_train_dense[:, col], ord=1) != 0:
    #         topk_train_dense[:, col] *= (1.0 / np.linalg.norm(topk_train_dense[:, col], ord=1))
    # topk_train = sp.csr_matrix(topk_train_dense)


    ''' Training: Set up model and train '''
    tf.reset_default_graph()
    tf.set_random_seed(0)
    
    if use_enc:
        model = DPGNN_enc(d, nc, FLAGS.hidden_size, FLAGS.nlayers, FLAGS.nEncLayers, FLAGS.nEncOut, FLAGS.lr, FLAGS.weight_decay, FLAGS.dropout, FLAGS.EncDropout,
                      FLAGS.clip_bound_sgd, FLAGS.sigma_sgd, FLAGS.microbatches, dp_sgd=FLAGS.dp_sgd, enc_dp_sgd=FLAGS.Enc_dp_sgd,
                      sparse_features=type(train_attr_matrix) is not np.ndarray)
        train = train_w_enc
    else:
        model = DPGNN(d, nc, FLAGS.hidden_size, FLAGS.nlayers, FLAGS.lr, FLAGS.weight_decay, FLAGS.dropout,
                      FLAGS.clip_bound_sgd, FLAGS.sigma_sgd, FLAGS.microbatches, dp_sgd=FLAGS.dp_sgd, sparse_features=type(train_attr_matrix) is not np.ndarray)

        train = train_solo

    sess = tf.Session()
    with sess.as_default():
        tf.global_variables_initializer().run()
        print("Training starts ... ")

        if use_enc:
            for epoch in tqdm(range(FLAGS.encoder_epochs)):
                random_index = np.random.permutation(len(train_labels))
                train_index = train_index[random_index]

                train_encoder(
                    sess=sess, model=model, attr_matrix=train_attr_matrix,
                    train_idx=train_index, topk_train=my_topk, labels=train_labels,
                    epoch=epoch, batch_size=FLAGS.batch_size)


        for epoch in tqdm(range(FLAGS.max_epochs)):
            random_index = np.random.permutation(len(train_labels))
            train_index = train_index[random_index]

            train(
                sess=sess, model=model, attr_matrix=train_attr_matrix,
                train_idx=train_index, topk_train=my_topk, labels=train_labels,
                epoch=epoch, batch_size=FLAGS.batch_size)

        print("Training finished.")
        # power iteration inference
        predictions = model.predict(
            sess=sess, adj_matrix=test_adj_matrix, attr_matrix=test_attr_matrix, alpha=FLAGS.alpha,
            nprop=FLAGS.nprop_inference, ppr_normalization=FLAGS.ppr_normalization)
        test_acc = accuracy_score(test_labels, predictions)
        print(f'''Testing accuracy: {test_acc:.4f}''')

        f = open("dp_experiment_out.txt", "a")
        f.write(f"dataset: {FLAGS.data_file}, GM: {FLAGS.dp_ppr}, EM:{FLAGS.EM}, V0:{FLAGS.report_val_eps}, DP-PPR epsilon: {epsilon}, DPSGD epsilon: {eps_sgd}, K: {FLAGS.topk}, Test acc: {test_acc:.4f}\n")
        f.close()




if __name__ == '__main__':
    app.run(main)
