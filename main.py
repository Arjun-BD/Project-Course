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
from pynverse import inversefunc
import networkx as nx
import scipy.sparse as sp
from node2vec import Node2Vec
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

# Set up logging
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
flags.DEFINE_integer('topk', 2 , 'Number of PPR neighbors for each node')
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
flags.DEFINE_integer('epsilon_label_dp', 3, 'Epsilon for Label DP')
flags.DEFINE_bool('label_dp', False, 'Use label DP or not')
flags.DEFINE_bool('feature_aware',False,'Activates feature aware PPR')
flags.DEFINE_bool('gravity',False,'Determines whether to use a gravity based clustering instead of PPR')
flags.DEFINE_bool('heat',False,'Determines whether to use a heat transfer    based clustering instead of PPR')

FLAGS = flags.FLAGS

def compute_Q(adj_matrix):
    
    if isinstance(adj_matrix, csr_matrix):
        adj_matrix = adj_matrix.toarray()
    G = nx.from_numpy_array(adj_matrix)
    N = len(adj_matrix)
    edges = G.number_of_edges()
    
    # Compute shortest path lengths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                R[i, j] = shortest_paths[i].get(j, np.inf)  # Use np.inf if no path exists
    
    # Compute closeness centrality
    CC = np.array([ (N-1) / np.sum(R[i]) if np.sum(R[i]) != 0 else 0 for i in range(N) ])
    
    # Compute network density
    density_G = (2 * edges) / (N * (N - 1))
    
    # Compute degree density
    degrees = np.array([G.degree(i) for i in range(N)])
    Dd = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if R[i, j] > 0 and R[i, j] != np.inf:
                Dd[i, j] = degrees[j] / (np.pi * R[i, j]**2)
    
    # Compute exponent term e^(EC(vi) - EC(vj))
    EC = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    exp_EC_diff = np.exp(EC[:, None] - EC[None, :])
    
    # Compute final Q values
    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and R[i, j] > 0 and R[i, j] != np.inf:
                Q[i, j] = (degrees[i] * exp_EC_diff[i, j] * density_G * Dd[i, j]) / R[i, j]
    
    return Q


def compute_node2vec_embeddings(adj_matrix, dim=16, walk_length=30, num_walks=200, p=1, q=1):

    # Convert sparse adjacency matrix to a NetworkX graph
    if sp.issparse(adj_matrix):  # If sparse
        G = nx.from_scipy_sparse_matrix(adj_matrix)
    else:  # If dense
        G = nx.from_numpy_array(adj_matrix)

    # Run Node2Vec
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings, ensuring node order matches adjacency matrix indices
    embeddings = np.array([model.wv[str(node)] for node in sorted(G.nodes())])

    return embeddings

def compute_gravitational_importance(adj_matrix, embeddings, G_const=1e-3):
    """Compute gravitational force-based node importance using adjacency matrix."""
    
    # Compute node degrees from adjacency matrix (supports sparse and dense formats)
    if isinstance(adj_matrix, csr_matrix):  # Sparse case
        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    else:  # Dense case
        degrees = np.sum(adj_matrix, axis=1)
    
    # Compute pairwise Euclidean distances between node embeddings
    dist_matrix = cdist(embeddings, embeddings, metric="euclidean")

    # Avoid division by zero
    dist_matrix[dist_matrix < 1e-6] = 1e-6  

    # Compute gravitational force: F_ij = G * (m_i * m_j) / d_ij^2
    gravity_matrix = G_const * np.outer(degrees, degrees) / (dist_matrix ** 2)
    
    return gravity_matrix



def main(unused_argv):
    tf.compat.v1.disable_v2_behavior()

    ''' Load Data '''
    start = time.time()

    FLAGS.microbatches = FLAGS.batch_size

    train_labels, train_adj_matrix, train_attr_matrix, train_index, test_labels, test_adj_matrix, \
    test_attr_matrix, test_index, num_nodes, num_class, num_attr, num_edges = utils.get_data(FLAGS.data_file,
                                                     privacy_amplify_sampling_rate=FLAGS.privacy_amplify_sampling_rate)
    
    print(train_labels.shape)
    temp = FLAGS.data_file.split('/')[1]

    np.savez(f"train_adj&train_attr{temp}.npz", 
         train_adj_matrix=train_adj_matrix.toarray(), 
         train_attr_matrix=train_attr_matrix.toarray(), 
         train_labels=train_labels)


    d = num_attr
    nc = num_class

    #-------------------Testing Label DP--------
    if FLAGS.label_dp:
        nc = len(set(train_labels))  # Number of unique classes (C)
        p_keep = np.exp(FLAGS.epsilon_label_dp) / (np.exp(FLAGS.epsilon_label_dp) + nc - 1)

        for i in range(len(train_labels)):
            if np.random.rand() > p_keep:  # Flip with probability 1 - p_keep
                new_label = np.random.choice([l for l in range(nc) if l != train_labels[i]])
                train_labels[i] = new_label  # Ensure different label
                
        print("done label dp")
    #-------------------------


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
    start = time.time()
    if(not FLAGS.ppr_normalization == 'cosine_similarity'):
        if(not FLAGS.gravity and not FLAGS.heat):
            topk_train = ppr.topk_ppr_matrix_ista(train_adj_matrix, FLAGS.alpha, FLAGS.eps, FLAGS.rho, train_index[:ppr_num],
                                                FLAGS.topk, FLAGS.sigma_ista, FLAGS.clip_bound_ista, FLAGS.dp_ppr,
                                                FLAGS.em_sensitivity, FLAGS.report_val_eps, FLAGS.EM, FLAGS.EM_eps,features = train_attr_matrix,Feature_Aware=FLAGS.feature_aware)
            print("--------------------------------------------------------------------------------------------------------------")
        elif(not FLAGS.gravity):
            topk_train = compute_Q(train_adj_matrix)
            
            topk_train = sp.csr_matrix(topk_train)
            print(topk_train)
        else:
            embeddings = compute_node2vec_embeddings(train_adj_matrix)
            topk_train = compute_gravitational_importance(train_adj_matrix,embeddings)
            threshold = 1e-4
            topk_train[topk_train < threshold] = 0

            topk_train = sp.csr_matrix(topk_train)
            

    else : topk_train = ppr.random_walk_with_cosine_similarity(adj_matrix=train_adj_matrix, adj_attr_matrix=train_attr_matrix, alpha=FLAGS.alpha, num_steps=20, nodes=train_index[:ppr_num], topk=FLAGS.topk)

    if ppr_num < len(train_index):
        topk_train_I = np.identity(len(train_index))
        topk_train_I = topk_train_I[ppr_num:]
        topk_train_dense = topk_train.toarray()
        topk_train_dense_full = np.concatenate((topk_train_dense, topk_train_I), axis=0)
        topk_train = sp.csr_matrix(topk_train_dense_full)

    time_preprocessing = time.time() - start

        
    print(f"Calculate Train PPR Matrix Runtime: {time_preprocessing:.2f}s")

    # # normalize l1 norm of each column of topk_train'''
    topk_train_dense = topk_train.toarray()
    for col in range(len(topk_train_dense[0, :])):
        if np.linalg.norm(topk_train_dense[:, col], ord=1) != 0:
            topk_train_dense[:, col] *= (1.0 / np.linalg.norm(topk_train_dense[:, col], ord=1))
    topk_train = sp.csr_matrix(topk_train_dense)


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
                    train_idx=train_index, topk_train=topk_train, labels=train_labels,
                    epoch=epoch, batch_size=FLAGS.batch_size)


        for epoch in tqdm(range(FLAGS.max_epochs)):
            random_index = np.random.permutation(len(train_labels))
            train_index = train_index[random_index]
            
            train(
                sess=sess, model=model, attr_matrix=train_attr_matrix,
                train_idx=train_index, topk_train=topk_train, labels=train_labels,
                epoch=epoch, batch_size=FLAGS.batch_size)

        print("Training finished.")
        # power iteration inference
        variables = tf.trainable_variables()
        layer_vars = sess.run(variables)
        param_dict = {var.name : sess.run(var) for var in variables}
        # print(param_dict)
        #save model
      
        # np.savez(f'model_{temp}_dpsgd_{FLAGS.dp_ppr}_sampling{FLAGS.privacy_amplify_sampling_rate * 100}_eps{epsilon}pct.npz', **param_dict)
        if FLAGS.gravity:
            embeddings = compute_node2vec_embeddings(test_adj_matrix)
            predictions = model.predict(
                sess=sess, adj_matrix=test_adj_matrix, attr_matrix=test_attr_matrix, alpha=FLAGS.alpha,
                nprop=FLAGS.nprop_inference, ppr_normalization=FLAGS.ppr_normalization,feature_aware=FLAGS.feature_aware,features=test_attr_matrix,gravity = True,embeddings=embeddings)
        else:

            predictions = model.predict(
                sess=sess, adj_matrix=test_adj_matrix, attr_matrix=test_attr_matrix, alpha=FLAGS.alpha,
                nprop=FLAGS.nprop_inference, ppr_normalization=FLAGS.ppr_normalization,feature_aware=FLAGS.feature_aware,features=test_attr_matrix,heat = FLAGS.heat)
        test_acc = accuracy_score(test_labels, predictions)
        print(f'''Testing accuracy: {test_acc:.4f}''')

        f = open("dp_experiment_out.txt", "a")

        print('Epsilon : ', epsilon)
        f.write(f"dataset: {FLAGS.data_file}, GM: {FLAGS.dp_ppr}, EM:{FLAGS.EM}, V0:{FLAGS.report_val_eps}, DP-PPR epsilon: {epsilon}, DPSGD epsilon: {eps_sgd}, K: {FLAGS.topk}, Test acc: {test_acc:.4f}\n")
        f.close() 
        



if __name__ == '__main__':
    app.run(main)