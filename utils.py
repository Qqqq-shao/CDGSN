import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve,auc
import sklearn.metrics as sm
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
import torch
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from warnings import filterwarnings
filterwarnings('ignore')
def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


def load_network_data(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    X[X > 0] = 1  # feature binarization
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    return A, X, Y
# def feature_compression(features, dim=200):
#     """Preprcessing of features"""
#     features = features.toarray()
#     feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
#     return feat


def my_scale_sim_mat(w):
    """L1 row norm of a matrix"""
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)
    return w

#
# def agg_tran_prob_mat(g, step):
#     """aggregated K-step transition probality"""
#     g = my_scale_sim_mat(g)
#     g = csc_matrix.toarray(g)
#     a_k = g
#     a = g
#     for k in np.arange(2, step+1):
#         a_k = np.matmul(a_k, g)
#         a = a+a_k/k
#     return a

#
# def compute_ppmi(a):
#     """compute PPMI, given aggregated K-step transition probality matrix as input"""
#     np.fill_diagonal(a, 0)
#     a = my_scale_sim_mat(a)
#     (p, q) = np.shape(a)
#     col = np.sum(a, axis=0)
#     col[col == 0] = 1
#     ppmi = np.log((float(p)*a)/col[None, :])
#     idx_nan = np.isnan(ppmi)
#     ppmi[idx_nan] = 0
#     ppmi[ppmi < 0] = 0
#     return ppmi

# def batch_ppmi(train_edge_index_s,train_edge_index_t, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
#     """return the PPMI matrix between nodes in each batch"""
#     # #proximity matrix between source network nodes in each mini-batch
#     # noinspection DuplicatedCode
#     a_s = np.zeros((int(train_edge_index_s), int(train_edge_index_s)))
#     for ii in range(int(train_edge_index_s)):
#         for jj in range(int(train_edge_index_s)):
#             if ii != jj:
#                 a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
#     # #proximity matrix between target network nodes in each mini-batch
#     # noinspection DuplicatedCode
#     a_t = np.zeros((int(train_edge_index_t), int(train_edge_index_t)))
#     for ii in range(int(train_edge_index_t)):
#         for jj in range(int(train_edge_index_t)):
#             if ii != jj:
#                 a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
#     return my_scale_sim_mat(a_s), my_scale_sim_mat(a_t)


# def shuffle_aligned_list(data):
#     num = data[0].shape[0]
#     shuffle_index = np.random.permutation(num)
#     return shuffle_index, [d[shuffle_index] for d in data]

def edge_prepare(filename):
    adj, features, labels = load_network_data(filename + '.mat')
    features = features / 1.0
    labels = labels / 1.0
    features_tensor = torch.FloatTensor(features)

    com_labels = np.matmul(labels, labels.transpose())
    com_labels[com_labels > 0] = 1
    com_labels[com_labels == 0] = -1

    adj2 = adj.toarray()
    row, col = np.diag_indices_from(adj2)
    adj2[row, col] = 0.0
    adj = csc_matrix(adj2)

    adj_up = sp.triu(adj, 0)
    com_labels_up = sp.triu(com_labels, 0)
    edge_labels = np.multiply(adj_up.toarray(), com_labels_up.toarray())

    x_pos, y_pos = np.where(edge_labels == 1)
    pos_edges = np.array(list(zip(x_pos, y_pos)))
    x_neg, y_neg = np.where(edge_labels == -1)
    neg_edges = np.array(list(zip(x_neg, y_neg)))

    adj_pos = edge_labels.copy()
    adj_pos = adj_pos + adj_pos.transpose()
    adj_pos[adj_pos == -1] = 0

    adj_pos = csc_matrix(adj_pos)


    return pos_edges,neg_edges,adj,adj_pos



def edge_rep_construct(node_rep, edge_label_random, edge_type):
    if edge_type == 'concat':
        edge_rep = torch.cat((node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :]), 1)
    elif edge_type == 'L1':
        edge_rep = torch.abs(node_rep[edge_label_random[:, 0], :] - node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'L2':
        edge_rep = torch.square(node_rep[edge_label_random[:, 0], :] - node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'had':
        edge_rep = torch.mul(node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'avg':
        edge_rep = torch.add(node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :]) / 2

    return edge_rep,node_rep[edge_label_random[:, 0], :],node_rep[edge_label_random[:, 1], :]

# def edge_PPMI_construct(PPMI, pos_edges, neg_edges, edge_type):
#     if edge_type == 'concat':
#         edge_rep_pos = torch.cat((PPMI[pos_edges[:, 0], :], PPMI[pos_edges[:, 1], :]), 1)
#         edge_rep_neg = torch.cat((PPMI[neg_edges[:, 0], :], PPMI[neg_edges[:, 1], :]), 1)
#     elif edge_type == 'L1':
#         edge_rep_pos = torch.abs(PPMI[pos_edges[:, 0], :] - PPMI[pos_edges[:, 1], :])
#         edge_rep_neg = torch.abs(PPMI[neg_edges[:, 0], :] - PPMI[neg_edges[:, 1], :])
#     elif edge_type == 'L2':
#         edge_rep_pos = torch.square(PPMI[pos_edges[:, 0], :] - PPMI[pos_edges[:, 1], :])
#         edge_rep_neg = torch.square(PPMI[neg_edges[:, 0], :] - PPMI[neg_edges[:, 1], :])
#     elif edge_type == 'had':
#         edge_rep_pos = torch.mul(PPMI[pos_edges[:, 0], :], PPMI[pos_edges[:, 1], :])
#         edge_rep_neg = torch.mul(PPMI[neg_edges[:, 0], :], PPMI[neg_edges[:, 1], :])
#     elif edge_type == 'avg':
#         edge_rep_pos = torch.add(PPMI[pos_edges[:, 0], :], PPMI[pos_edges[:, 1], :]) / 2
#         edge_rep_neg = torch.add(PPMI[neg_edges[:, 0], :], PPMI[neg_edges[:, 1], :]) / 2
#     edge_rep_pos2 = torch.abs(PPMI[pos_edges[:, 0], :] - PPMI[pos_edges[:, 1], :])
#     edge_rep_neg2 = torch.abs(PPMI[neg_edges[:, 0], :] - PPMI[neg_edges[:, 1], :])
#
#     # 将pos和neg的边的特征向量，拼接：
#     edge_rep = torch.cat((edge_rep_pos, edge_rep_neg,edge_rep_pos2,edge_rep_neg2), 0)
#
#     return edge_rep
def GET_AUC_ROC(true_label,predict):
    preds = torch.sigmoid(predict)
    preds=preds.cpu().detach().numpy()
    auc_roc=sm.roc_auc_score(true_label.cpu(),preds)
    return auc_roc


def GET_AUC_PR(y_true, y_logits):
    y_prob=torch.sigmoid(y_logits.cpu()).detach().numpy()
    y_prob=1-y_prob
    y_true=1-y_true
    precision, recall, thresholds = precision_recall_curve(y_true.cpu(), y_prob) # calculate precision-recall curve
    AUC_PR = auc(recall, precision)
    return AUC_PR


# def train_val_test(Y, seed, trp, vap):
#     np.random.seed(seed)
#     random_node_indices = np.random.permutation(Y.shape[0])
#     training_size = int(len(random_node_indices) * trp)
#     val_size = int(len(random_node_indices) * vap)
#     train_node_indices = random_node_indices[:training_size]
#     val_node_indices = random_node_indices[training_size:training_size + val_size]
#     test_node_indices = random_node_indices[training_size + val_size:]
#     return train_node_indices, val_node_indices, test_node_indices

# def normalize(mx):
#     """Symmetrically normalize adjacency matrix."""
#     mx = sp.coo_matrix(mx)
#     rowsum = np.array(mx.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)
# def f1_scores(y_pred, y_true):
#     def predict(y_tru, y_pre):
#         top_k_list = np.array(np.sum(y_tru, 1), np.int32)
#         prediction = []
#         for i in range(y_tru.shape[0]):
#             pred_i = np.zeros(y_tru.shape[1])
#             pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
#             prediction.append(np.reshape(pred_i, (1, -1)))
#         prediction = np.concatenate(prediction, axis=0)
#         return np.array(prediction, np.int32)
#     results = {}
#     predictions = predict(y_true, y_pred)
#     averages = ["micro", "macro"]
#     for average in averages:
#         results[average] = f1_score(y_true, predictions, average=average)
#     return results["micro"], results["macro"]

#
# def net_pro_loss(emb, a):
#     r = torch.sum(emb*emb, 1)
#     r = torch.reshape(r, (-1, 1))
#     dis = r-2*torch.matmul(emb, emb.T)+r.T
#     return torch.mean(torch.sum(a.__mul__(dis), 1))

# def MSELOSS(emb,adj,adj_penalty):
#
#     emb=F.normalize(emb)
#     reconstructed_adj=torch.mm(emb,emb.t())
#     reconstructed_adj=torch.tanh(reconstructed_adj)
#     # reconstructed_adj=torch.mul(reconstructed_adj,adj_penalty)
#     loss_func = torch.nn.MSELoss()
#     loss = loss_func(reconstructed_adj,adj)
#     return loss
#
#


# def random_edge(pos_edges_s, labels_pos_s,neg_edges_s, labels_neg_s,pos_edges_t, labels_pos_t,neg_edges_t, labels_neg_t,random_state):
#
#     np.random.seed(random_state)
#
#     edge_label_s = np.vstack((np.hstack((pos_edges_s, labels_pos_s)), np.hstack((neg_edges_s, labels_neg_s))))
#     edge_label_t = np.vstack((np.hstack((pos_edges_t, labels_pos_t)), np.hstack((neg_edges_t, labels_neg_t))))
#     edge_label_s_random = np.random.permutation(edge_label_s)
#     edge_label_t_random = np.random.permutation(edge_label_t)
#     edge_label_s_random_tensor = torch.FloatTensor(edge_label_s_random)
#     edge_label_t_random_tensor = torch.FloatTensor(edge_label_t_random)
#     return edge_label_s_random,edge_label_t_random,edge_label_s_random_tensor,edge_label_t_random_tensor

def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    """calculate inner product"""
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

# def inter_view_nei_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
#     """inter view neighbor contrastive loss"""
#     adj[adj > 0] = 1
#
#     f = lambda x: torch.exp(x / tau)
#     between_sim = f(sim(z1, z2, hidden_norm))
#
#     nei_count = torch.sum(adj, 1) + 1
#     nei_count = torch.squeeze(torch.tensor(nei_count))
#
#     loss = (between_sim.mul(adj)).sum(1) / between_sim.sum(1)
#     loss = loss / nei_count
#
#     return -torch.log(loss + 1e-10)


def nei_dis_loss1(z1: torch.Tensor, z2: torch.Tensor, tau, adj, hidden_norm: bool = True):
    """neighbor discrimination contrastive loss"""
    ###先求和再log
    # np.fill_diagonal(adj, 0) #remove self-loop
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    adj[adj > 0] = 1
    # nei_count=np.sum(adj,1)*2+1 ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.sum(adj, 1) * 2 + 1  ###intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))
    # adj = torch.tensor(adj)

    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1, hidden_norm))
    between_sim = f(sim(z1, z2, hidden_norm))

    loss = (between_sim.diag() + (refl_sim.mul(adj)).sum(1) + (between_sim.mul(adj)).sum(1)) / (
            refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
    loss = loss / nei_count  ###divided by the number of positive pairs for each node

    return -torch.log(loss)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, adj,
                     mean: bool = True, tau: float = 1.0, hidden_norm: bool = True):
    h1 = z1
    h2 = z2

    l1 = nei_dis_loss1(h1, h2, tau, adj, hidden_norm)
    l2 = nei_dis_loss1(h2, h1, tau, adj, hidden_norm)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret


# def multihead_contrastive_loss(heads, adj, tau: float = 1.0):
#     """multi-head contrastive loss"""
#     ###算每个head到第一个head
#     loss = torch.tensor(0, dtype=float, requires_grad=True)
#     for i in range(1, len(heads)):
#         loss = loss + contrastive_loss(heads[0], heads[i], adj, tau=tau)
#     return loss / (len(heads) - 1)


def my_scale_sim_mat_torch(w):
    """L1 row norm of a matrix of torch version"""
    r = 1 / w.sum(1)
    r[r.isinf()] = 0
    return r.diag() @ w

def calculate_centroid_2(label, emb):
    """calculate centroid of each class"""
    norm_Y = my_scale_sim_mat_torch(label.T)
    centroid = torch.mm(norm_Y, emb)
    return centroid
def one_hot_encode_torch(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    x = x.type(torch.LongTensor)
    return torch.eye(n_classes)[x]

def class_class_inter_view_nei_loss_1(z1: torch.Tensor, z2: torch.Tensor, tau, O_inter,
                                      hidden_norm: bool = True):
    """class-class inter view neighbor contrastive loss"""

    f = lambda x: torch.exp(x / tau)

    inter_sim_st = f(sim(z1, z2, hidden_norm))

    molecule = inter_sim_st.diag() * O_inter.diag()
    denominator = (inter_sim_st * O_inter).sum(1)
    # 分母为0，会导致nan
    denominator[denominator == 0] = 1
    loss = molecule / denominator
    loss[loss == 0] = 1
    # loss[loss.isnan()] = 1
    return -torch.log(loss) 


# def class_class_inter_view_nei_loss_2(z1: torch.Tensor, z2: torch.Tensor, tau, O_inter, O_intra,
#                                       hidden_norm: bool = True):
#     """class-class inter view neighbor contrastive loss"""
#     O_intra = O_intra - (O_intra.diag().diag())
#
#     f = lambda x: torch.exp(x / tau)
#
#     inter_sim_st = f(sim(z1, z2, hidden_norm))
#     intra_sim_ss = f(sim(z1, z1, hidden_norm))
#
#     molecule = inter_sim_st.diag() * O_inter.diag()
#     denominator = (inter_sim_st * O_inter).sum(1) + (intra_sim_ss * O_intra).sum(1)
#     # 分母为0，会导致nan
#     denominator[denominator == 0] = 1
#     loss = molecule / denominator
#     loss[loss == 0] = 1
#     # loss[loss.isnan()] = 1
#     return -torch.log(loss)


def calculate_pred_label_t(pred_label, cluster_label):
    """calculate predicted label by comparing clf loss with pred label kmean"""
    # _, indices = torch.max(pred_logit_s, dim=1)
    # pred_logit_s = one_hot_encode_torch(indices, pred_logit_t.shape[1])
    # pred_logit_s = pred_logit_s.to(pred_logit_t.device)
    return pred_label * cluster_label
