from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import random
import torch
from torch import nn
import dgl
import torch.nn.functional as F
import utils
from model import CDGSN
from flip_gradient import GradReverse
from utils import GET_AUC_PR, GET_AUC_ROC, edge_prepare, normalize_features

from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr-ini', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--l2-w', type=float, default=0.0005,
                    help='weight of L2-norm regularization')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--Hid-dim-edge-clf', type=int, default=128,
                    help='Number of hidden edge clf layers.')
parser.add_argument('--Hid-dim-node-clf', type=int, default=32,
                    help='Number of hidden node clf layers.')
parser.add_argument('--Hid-dim-edge-domain-1', type=int, default=128,
                    help='Number of hidden edge clf layers.')
parser.add_argument('--Hid-dim-edge-domain-2', type=int, default=32,
                    help='Number of hidden edge clf layers.')
parser.add_argument("--num-heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=8,
                    help="number of hidden layers")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-hidden", type=int, default=64,
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.3,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--edge-type', type=str, default='concat',
                    help='concat, had, avg')
parser.add_argument('--source', type=str, default='acmv9',
                    help='source network')
parser.add_argument('--target', type=str, default='citationv1',
                    help='target network')
parser.add_argument("--tau", type=float, default=1, help="temperature-scales")
parser.add_argument('--Weight-ada', type=float, default=0.1,
                    help="Weight of adversarial domain adaptation loss")
parser.add_argument('--Weight-node-clf', type=float, default=1,
                    help="Weights of node classification loss")
parser.add_argument('--Weight-edge-clf', type=float, default=1,
                    help="Weights of edge classification loss")
parser.add_argument('--Weight-cda', type=float, default=1,
                    help="Weight of class-aware contrastive domain adaptation loss")
parser.add_argument('--Weight-SelPNL', type=float, default=0.1,
                    help="Weight of positive and negative self-training loss")
parser.add_argument('--threshold_pos', type=float, default=0.8,
                    help="Positive pseudo labeling threshold")
parser.add_argument('--threshold_neg', type=float, default=0.1,
                    help="Negative pseudo labeling threshold")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

numRandom = 5 # number of random splits
source = args.source
target = args.target
edge_type = args.edge_type

emb_filename = str(source) + '_' + str(target)
cost_sensitive = False  ##whether give higher weight to scarce class in CE Loss

f = open('./output/' + 'CDGSN_' + emb_filename + '_' + edge_type + '.txt', 'a')
f.write('{}\n'.format(args))
f.flush()

A_s, X_s, Y_s = utils.load_network(str(source) + '.mat')
features_s = normalize_features(X_s.todense())
features_s = torch.Tensor(features_s)
Y_s_node_label = np.argmax(Y_s, 1).reshape(-1, 1)
Y_s_node_label_dict = {}
for i in range(features_s.size()[0]):
    key = Y_s_node_label[i][0]
    Y_s_node_label_dict.setdefault(key, []).append(i)
Y_s = torch.Tensor(Y_s)
num_feats = X_s.shape[1]
A_t, X_t, Y_t = utils.load_network(str(target) + '.mat')
Y_t = torch.Tensor(Y_t)
num_nodes_T = X_t.shape[0]
features_t = normalize_features(X_t.todense())
features_t = torch.Tensor(features_t)

##get indices of positive and negative edges

pos_edges_s, neg_edges_s, _, _ = edge_prepare(source)
pos_edges_t, neg_edges_t, _, _ = edge_prepare(target)

labels_pos_s = np.ones((pos_edges_s.shape[0], 1))
labels_neg_s = np.zeros((neg_edges_s.shape[0], 1))
labels_s = np.vstack((labels_pos_s, labels_neg_s))

labels_pos_t = np.ones((pos_edges_t.shape[0], 1))
labels_neg_t = np.zeros((neg_edges_t.shape[0], 1))
labels_t = np.vstack((labels_pos_t, labels_neg_t))


random_state = 0

n_input_s = X_s.shape[1]
n_input_t = X_t.shape[1]

xb_s = torch.FloatTensor(X_s.tocsr().toarray())
xb_t = torch.FloatTensor(X_t.tocsr().toarray())

domain_label_edge = np.vstack([np.tile([1., 0.], [labels_s.shape[0], 1]),
                               np.tile([0., 1.], [labels_t.shape[0], 1])])  # [1,0] for source, [0,1] for target
domain_label_node = np.vstack([np.tile([1., 0.], [X_s.shape[0], 1]), np.tile([0., 1.], [X_t.shape[0], 1])])

last_auc_roc_trp = []

last_AP_neg_trp = []

weit_p = pos_edges_s.shape[0] / neg_edges_s.shape[0]
g_s = dgl.from_scipy(A_s)
g_s = dgl.remove_self_loop(g_s)
g_s = dgl.add_self_loop(g_s)

g_t = dgl.from_scipy(A_t)
g_t = dgl.remove_self_loop(g_t)
g_t = dgl.add_self_loop(g_t)
if args.gpu < 0:
    cuda = False
else:
    cuda = True
    g_s = g_s.int().to(args.gpu)
    g_t = g_t.int().to(args.gpu)
    Y_t = Y_t.to(args.gpu)
    Y_s = Y_s.to(args.gpu)

heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
mlp_input_dim = int((args.num_hidden) * (args.num_heads))
while random_state < numRandom:

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)
    random.seed(random_state)
    edge_label_s = np.vstack((np.hstack((pos_edges_s, labels_pos_s)), np.hstack((neg_edges_s, labels_neg_s))))
    edge_label_t = np.vstack((np.hstack((pos_edges_t, labels_pos_t)), np.hstack((neg_edges_t, labels_neg_t))))
    edge_label_s_random = np.random.permutation(edge_label_s)
    edge_label_t_random = np.random.permutation(edge_label_t)
    edge_label_s_random_tensor = torch.FloatTensor(edge_label_s_random)
    edge_label_t_random_tensor = torch.FloatTensor(edge_label_t_random)

    random_state = random_state + 1

    print('%d-th random split' % (random_state))

    if edge_type == 'concat':

        model = CDGSN(
            num_layers=args.num_layers,
            in_dim=num_feats,
            num_hidden=args.num_hidden,
            heads=heads,
            activation=F.elu,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            dropout=args.dropout,
            input_dim_edge=mlp_input_dim * 2,
            input_dim_node=mlp_input_dim,
            hid_dim_edge_clf=args.Hid_dim_edge_clf,
            hid_dim_node_clf=args.Hid_dim_node_clf,
            Hid_dim_edge_domain_1=args.Hid_dim_edge_domain_1,
            Hid_dim_edge_domain_2=args.Hid_dim_edge_domain_2

        )


    else:
        model = CDGSN(
            num_layers=args.num_layers,
            in_dim=num_feats,
            num_hidden=args.num_hidden,
            heads=heads,
            activation=F.elu,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            dropout=args.dropout,
            input_dim_edge=mlp_input_dim,
            input_dim_node=mlp_input_dim,
            hid_dim_edge_clf=args.Hid_dim_edge_clf,
            hid_dim_node_clf=args.Hid_dim_node_clf,
            Hid_dim_edge_domain_1=args.Hid_dim_edge_domain_1,
            Hid_dim_edge_domain_2=args.Hid_dim_edge_domain_2

        )
    if cuda:
        model.cuda()
        features_s = features_s.cuda()
        features_t = features_t.cuda()
        edge_label_s_random_tensor = edge_label_s_random_tensor.cuda()
        edge_label_t_random_tensor = edge_label_t_random_tensor.cuda()

    clf_loss_f = F.binary_cross_entropy_with_logits
    cf_loss_f_node = nn.BCEWithLogitsLoss(reduction='none')
    domain_loss_f = nn.CrossEntropyLoss()


    train_loss_all = []
    train_auc_roc_s_all = []
    test_auc_roc_t_all = []
  

    total_loss_all = []
    AP_s_neg_all = []
    AP_t_neg_all = []
    t_total = time.time()

    pred_label_t=torch.zeros((Y_t.shape[0],5))
    for cEpoch in range(args.epochs):
        t = time.time()
        p = float(cEpoch) / args.epochs
        lr = args.lr_ini / (1. + 10 * p) ** 0.75
        grl_lambda = (2. / (1. + np.exp(-10. * p)) - 1) # gradually change from 0 to 1
        GradReverse.rate = grl_lambda
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_w)
        model.train()
        optimizer.zero_grad()
        pred_logit_s, pred_logit_t, pred_logit_node_s, pred_logit_node_t, d_logit, emb_node_s, emb_node_t = model(features_s, features_t, g_s, g_t, edge_label_s_random,edge_label_t_random,edge_type)
        d_prob = F.softmax(d_logit)


        pred_logit_node_t_prob = F.softmax(pred_logit_node_t)


        label_s_sum = (Y_s.sum(0) > 0).float().to(args.gpu)
        label_t_sum = (pred_label_t.sum(0) > 0).float().to(args.gpu)
        O_ST = label_s_sum.view(-1, 1) * label_t_sum.view(1, -1)
        O_SS = label_s_sum.view(-1, 1) * label_s_sum.view(1, -1)
        O_TT = label_t_sum.view(-1, 1) * label_t_sum.view(1, -1)
        pred_label_t = pred_label_t.to(args.gpu)

        centroid_s = utils.calculate_centroid_2(Y_s, emb_node_s)
        centroid_t = utils.calculate_centroid_2(pred_label_t, emb_node_t)

        cross_view_st = utils.class_class_inter_view_nei_loss_1(centroid_s, centroid_t, args.tau, O_ST)


        cross_network_constrastive_loss = cross_view_st
        cross_network_constrastive_loss = cross_view_st.mean()
        _, indices = torch.max(pred_logit_node_t, dim=1)
        loss_ce = 0
        loss_nl = 0


        pred_logit_node_t_softmax = F.softmax(pred_logit_node_t, dim=1)
        positive_idx = np.array(torch.where(pred_logit_node_t_softmax > args.threshold_pos)[0].cpu())
        if positive_idx.size > 0:
            # positive learning
            loss_ce += F.cross_entropy(pred_logit_node_t[positive_idx], indices[positive_idx], reduction='mean')
        for i in range(0,5):
            nl_idx = np.array(torch.where(pred_logit_node_t_softmax[:, i] < args.threshold_neg)[0].cpu())

            if nl_idx.size > 0:
                nl_logits = pred_logit_node_t[nl_idx]
                pred_nl = F.softmax(nl_logits, dim=1)
                pred_nl = 1 - pred_nl
                pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
                indices_nl_mask = indices[nl_idx]
                y_nl = torch.ones((nl_logits.shape)) #.to(device=args.device, dtype=logits.dtype)
                # negative learning
                loss_nl +=-torch.sum(torch.log(pred_nl[:,i]) , dim = -1)/(nl_idx.size  + 1e-7)


        if cost_sensitive:  ##give higher weight to scarce class
            clf_loss_edge = clf_loss_f(pred_logit_s, edge_label_s_random_tensor[:, 2].reshape(-1, 1), weight=weit_p)
            clf_loss_node = cf_loss_f_node(pred_logit_node_s, Y_s)
            clf_loss_node = torch.sum(clf_loss_node) / np.sum(Y_s.shape[0])
        else:
            clf_loss_edge = clf_loss_f(pred_logit_s, edge_label_s_random_tensor[:, 2].reshape(-1, 1))
            clf_loss_node = cf_loss_f_node(pred_logit_node_s, Y_s)
            clf_loss_node = torch.sum(clf_loss_node) / np.sum(Y_s.shape[0])

        domain_loss_edge = domain_loss_f(d_logit, torch.argmax(torch.FloatTensor(domain_label_edge).to(
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), 1))

        total_loss =  domain_loss_edge * args.Weight_ada + clf_loss_edge * args.Weight_edge_clf+args.Weight_cda*cross_network_constrastive_loss +(loss_nl+loss_ce)*args.Weight_SelPNL + clf_loss_node * args.Weight_node_clf
        total_loss.backward()
        total_loss_all.append(total_loss.item())

        optimizer.step()
        model.eval()  # deactivates dropout during validation run.
        with torch.no_grad():
            GradReverse.rate = 1.0

            pred_logit_xs, pred_logit_xt, pred_logit_node_xs, pred_logit_node_xt, _, emb_node_s, emb_node_t= model(
                features_s, features_t, g_s, g_t, edge_label_s_random,edge_label_t_random, edge_type)
            _, indices = torch.max(pred_logit_node_xt, dim=1)
            pred_label_clf = utils.one_hot_encode_torch(indices, pred_logit_node_xt.shape[1])
            norm_Y_s = utils.my_scale_sim_mat(np.array(Y_s.cpu()).T)
            norm_Y_s = torch.FloatTensor(norm_Y_s)
            emb_s_class = torch.mm(norm_Y_s, emb_node_s.cpu()) 
            kmeans = KMeans(n_clusters=5, init=emb_s_class, random_state=0).fit(emb_node_s.cpu())
            pred_Y_t_kmean = kmeans.predict(emb_node_t.cpu())
            pred_Y_t_kmean = np.eye(5)[pred_Y_t_kmean]
            pred_label_t = utils.calculate_pred_label_t(pred_label_clf, pred_Y_t_kmean).float()
            pred_label_t=pred_label_t.to(args.gpu)

        pred_prob_s = F.softmax(pred_logit_xs)
        pred_prob_t = F.softmax(pred_logit_xt)

        auc_pr_train_s = GET_AUC_PR(edge_label_s_random_tensor[:, 2].reshape(-1, 1),pred_logit_xs)
        auc_roc_train_s = GET_AUC_ROC(edge_label_s_random_tensor[:, 2].reshape(-1, 1), pred_logit_xs)

        auc_roc_test_t = GET_AUC_ROC(edge_label_t_random_tensor[:, 2].reshape(-1, 1), pred_logit_xt)
        auc_pr_test_t = GET_AUC_PR(edge_label_t_random_tensor[:, 2].reshape(-1, 1), pred_logit_xt)

        AP_s_pos = average_precision_score(edge_label_s_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           pred_logit_xs.cpu().detach().numpy(), pos_label=1)
        AP_t_pos = average_precision_score(edge_label_t_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           pred_logit_xt.cpu().detach().numpy(), pos_label=1)

        AP_s_neg = average_precision_score(1 - edge_label_s_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           1 - pred_logit_xs.cpu().detach().numpy(), pos_label=1)
        AP_t_neg = average_precision_score(1 - edge_label_t_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           1 - pred_logit_xt.cpu().detach().numpy(), pos_label=1)

        train_auc_roc_s_all.append(auc_roc_train_s)
        test_auc_roc_t_all.append(auc_roc_test_t)
        AP_s_neg_all.append(AP_s_neg)
        AP_t_neg_all.append(AP_t_neg)

        if (cEpoch + 1) % 10 == 0 or cEpoch == 0:
            print('Epoch: {:04d}'.format(cEpoch + 1),
                  'total_loss: {:.4f}'.format(total_loss.item()),
                  'Source auc_roc: {:.4f}'.format(auc_roc_train_s.item()),
                  'Target testing auc_roc: {:.4f}'.format(auc_roc_test_t.item()),
                  'Source AP_neg: {:.4f}'.format(AP_s_neg.item()),
                  'Target AP_neg: {:.4f}'.format(AP_t_neg.item()),
                  'time: {:.4f}s'.format(time.time() - t))

 
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("The last testing auc_roc %f and AP_neg % f " % (auc_roc_test_t, AP_t_neg))
  
    f.write('%d-th random initialization, the last testing auc_roc %f and AP_neg % f  \n' % (
        random_state, auc_roc_test_t, AP_t_neg))
    f.flush()
  
    last_auc_roc_trp.append(auc_roc_test_t)
    last_AP_neg_trp.append(AP_t_neg)


last_avg_auc_roc = np.mean(last_auc_roc_trp)
last_std_auc_roc = np.std(last_auc_roc_trp)


last_avg_AP_neg = np.mean(last_AP_neg_trp)
last_std_AP_neg = np.std(last_AP_neg_trp)

print('avg last testing auc_roc over %d random initialization: %f +/- %f' % (
    numRandom, last_avg_auc_roc, last_std_auc_roc))

print(
    'avg last testing AP_neg over %d random initialization: %f +/- %f' % (numRandom, last_avg_AP_neg, last_std_AP_neg))


f.write('avg last testing auc_roc over %d random splits: %f +/- %f  \n' % (
    numRandom, last_avg_auc_roc, last_std_auc_roc))
f.flush()

f.write('avg last testing AP_neg over %d random splits: %f +/- %f  \n' % (
    numRandom, last_avg_AP_neg, last_std_AP_neg))
f.flush()

f.close()