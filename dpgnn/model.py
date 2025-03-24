import numpy as np
import tensorflow.compat.v1 as tf
from . import tf_utils
from . import utils
from .privacy_utils import dp_optimizer
from dpgnn.utils import SparseRowIndexer
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import networkx as nx

class DPGNN:
    def __init__(self, d, nc, hidden_size, nlayers, lr,
                 weight_decay, dropout, clip_bound, sigma, microbatches, dp_sgd=False, sparse_features=True):
        self.nc = nc
        self.sparse_features = sparse_features

        if sparse_features:
            self.batch_feats = tf.sparse_placeholder(tf.float32, None, 'features')
        else:
            self.batch_feats = tf.placeholder(tf.float32, [None, d], 'features')
        self.batch_pprw = tf.placeholder(tf.float32, [None], 'ppr_weights')
        self.batch_idx = tf.placeholder(tf.int32, [None], 'idx')
        self.batch_labels = tf.placeholder(tf.int32, [None], 'labels')

        Ws = [tf.get_variable('W1', [d, hidden_size])]
        for i in range(nlayers - 2):
            Ws.append(tf.get_variable(f'W{i + 2}', [hidden_size, hidden_size]))

        feats_drop = tf_utils.mixed_dropout(self.batch_feats, dropout)
        if sparse_features:
            h = tf.sparse.sparse_dense_matmul(feats_drop, Ws[0])
        else:
            h = tf.matmul(feats_drop, Ws[0])
        for W in Ws[1:]:
            h = tf.nn.relu(h)
            h_drop = tf.nn.dropout(h, rate=dropout)
            h = tf.matmul(h_drop, W)

        self.embedding = h
        Wo = tf.get_variable(f'W{nlayers}', [hidden_size, nc])
        h = tf.nn.relu(self.embedding)
        h_drop = tf.nn.dropout(h, rate=dropout)
        h = tf.matmul(h_drop, Wo)

        self.logits = h

        self.weighted_logits = tf.tensor_scatter_nd_add(tf.zeros((tf.shape(self.batch_labels)[0], nc)),
                                                   self.batch_idx[:, None],
                                                   self.logits * self.batch_pprw[:, None])

        self.preds = tf.argmax(self.weighted_logits, 1)

        loss_per_node = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels,
                                                                       logits=self.weighted_logits)

        if dp_sgd:
            noise_multiplier = sigma / clip_bound
            self.loss = loss_per_node
            self.update_op = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=clip_bound,
                noise_multiplier=noise_multiplier,
                num_microbatches=microbatches,
                learning_rate=lr).minimize(self.loss)
        else:
            l2_reg = tf.add_n([tf.nn.l2_loss(weight) for weight in Ws])
            self.loss = tf.reduce_mean(loss_per_node) + weight_decay * l2_reg
            self.update_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.cached_train = {}
        self.cached_test = {}

    def feed_for_batch_train(self, attr_matrix, ppr_matrix, labels, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, labels)
        else:
            if key in self.cached_train:
                return self.cached_train[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, labels)
                self.cached_train[key] = feed
                return feed

    def feed_for_batch_test(self, attr_matrix, ppr_matrix, labels, key=None):
        if key is None:
            return self.gen_feed(attr_matrix, ppr_matrix, labels)
        else:
            if key in self.cached_test:
                return self.cached_test[key]
            else:
                feed = self.gen_feed(attr_matrix, ppr_matrix, labels)
                self.cached_test[key] = feed
                return feed

    def gen_feed(self, attr_matrix, ppr_matrix, labels):
        source_idx, neighbor_idx = ppr_matrix.nonzero()
        
        # Ensure source_idx and neighbor_idx are np.int32
        source_idx = source_idx.astype(np.int32)
        neighbor_idx = neighbor_idx.astype(np.int32)

        batch_attr = attr_matrix[neighbor_idx]
        feed = {
            self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
            self.batch_pprw: ppr_matrix[source_idx, neighbor_idx].A1,
            self.batch_labels: labels,
            self.batch_idx: source_idx,
        }
        return feed

    def gen_embed_feed(self, attr_matrix, train_index):
        batch_attr = attr_matrix[train_index]
        feed = {
            self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr,
        }
        return feed

    def _get_logits(self, sess, attr_matrix, nnodes, batch_size_logits=10000):
        logits = []
        for i in range(0, nnodes, batch_size_logits):
            batch_attr = attr_matrix[i:i + batch_size_logits]
            logits.append(sess.run(self.logits,
                                   {self.batch_feats: utils.sparse_feeder(batch_attr) if self.sparse_features else batch_attr}
                                   ))
        logits = np.row_stack(logits)
        return logits
    
    def compute_Q(self,adj_matrix):
        
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

    def compute_gravitational_importance(self,adj_matrix, embeddings, G_const=1e-3):
        """Compute gravitational force-based node importance using adjacency matrix."""
        
        print("ADJ!!!!!!",adj_matrix,type(adj_matrix))
        if isinstance(adj_matrix, sp.csr_matrix):
            degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        elif isinstance(adj_matrix, np.ndarray):
            if adj_matrix.ndim == 2:
                degrees = np.sum(adj_matrix, axis=1)
            else:
                raise ValueError("adj_matrix must be a 2D array")
        else:
            raise TypeError("adj_matrix must be a NumPy array or scipy sparse matrix")
        
        # Compute pairwise Euclidean distances between node embeddings
        dist_matrix = cdist(embeddings, embeddings, metric="euclidean")

        # Avoid division by zero
        dist_matrix[dist_matrix < 1e-6] = 1e-6  

        # Compute gravitational force: F_ij = G * (m_i * m_j) / d_ij^2
        gravity_matrix = G_const * np.outer(degrees, degrees) / (dist_matrix ** 2)

        # Thresholding to remove small values
        gravity_matrix[gravity_matrix < 1e-4] = 0

        return sp.csr_matrix(gravity_matrix)  # Return sparse matrix for efficiency

    
    def predict(self, sess, adj_matrix, attr_matrix, alpha, features=[],
            nprop=2, ppr_normalization='sym', gravity=False, batch_size_logits=10000,
            feature_aware=False, embeddings=[],heat = False):
    
        local_logits = self._get_logits(sess, attr_matrix, adj_matrix.shape[0], batch_size_logits)
        logits = local_logits.copy()

        if heat:
            Q_matrix = self.compute_Q(adj_matrix)
            Q_matrix = csr_matrix(Q_matrix)
            Q_matrix = Q_matrix.multiply(1 / Q_matrix.sum(axis=1).A1[:, None])  # Normalize
            
            for _ in range(nprop):
                logits = (1 - alpha) * (Q_matrix @ logits) + alpha * local_logits

        if gravity and len(embeddings) > 0:
            gravity_matrix = self.compute_gravitational_importance(adj_matrix=adj_matrix,embeddings=embeddings)
            gravity_matrix = gravity_matrix.multiply(1 / gravity_matrix.sum(axis=1).A1[:, None])  # Normalize
            
            for _ in range(nprop):
                logits = (1 - alpha) * (gravity_matrix @ logits) + alpha * local_logits
        
        elif ppr_normalization == 'sym':
            deg = adj_matrix.sum(1).A1
            deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
            for _ in range(nprop):
                if feature_aware:
                    feature_norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
                    feature_sim = (features @ features.T) / (feature_norm @ feature_norm.T)
                    feature_sim = feature_sim / feature_sim.sum(axis=1, keepdims=True)
                    logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) \
                            + alpha * (feature_sim @ local_logits)
                else:
                    logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) \
                            + alpha * local_logits
        
        elif ppr_normalization == 'col':
            deg_col = adj_matrix.sum(0).A1
            deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
            for _ in range(nprop):
                logits = (1 - alpha) * (adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
        
        elif ppr_normalization == 'row':
            deg_row = adj_matrix.sum(1).A1
            deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
            for _ in range(nprop):
                logits = deg_row_inv_alpha[:, None] * (adj_matrix @ logits) + alpha * local_logits
        
        elif ppr_normalization == 'cosine_similarity':
            attr_norms = np.linalg.norm(attr_matrix.toarray(), axis=1)
            attr_norms[attr_norms == 0] = 1e-12
            cosine_sim_matrix = attr_matrix @ attr_matrix.T
            cosine_sim_matrix /= attr_norms[:, None]
            cosine_sim_matrix /= attr_norms[None, :]
            transition_matrix = adj_matrix.multiply(cosine_sim_matrix)
            transition_matrix = sp.diags(1 / transition_matrix.sum(axis=1).A1) @ transition_matrix
            transition_matrix = -1 * transition_matrix
            for _ in range(nprop):
                logits = (1 - alpha) * (transition_matrix.T @ logits) + alpha * local_logits
        
        else:
            raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
        
        predictions = logits.argmax(1)
        return predictions
    def get_vars(self, sess):
        return sess.run(tf.trainable_variables())

    def set_vars(self, sess, new_vars):
        set_all = [
                var.assign(new_vars[i])
                for i, var in enumerate(tf.trainable_variables())]
        sess.run(set_all)


def train(sess, model, attr_matrix, train_idx, topk_train, labels, epoch, batch_size):
    if sp.issparse(attr_matrix):
        attr_matrix = SparseRowIndexer(attr_matrix)

    for i in range(0, len(train_idx), batch_size):
        if (i + batch_size) <= len(labels):
            feed_train = model.feed_for_batch_train(attr_matrix,
                                                    topk_train[train_idx[i:i + batch_size]],
                                                    labels[train_idx[i:i + batch_size]],
                                                    key=i)
            _, preds = sess.run([model.update_op, model.preds], feed_train)

        else:
            feed_train = model.feed_for_batch_train(attr_matrix,
                                                    topk_train[train_idx[len(labels)-batch_size:len(labels)]],
                                                    labels[train_idx[len(labels)-batch_size:len(labels)]],
                                                    key=i)

            _, preds = sess.run([model.update_op, model.preds], feed_train)

    return


def train_encoder(**kwargs):
    raise NotImplementedError
