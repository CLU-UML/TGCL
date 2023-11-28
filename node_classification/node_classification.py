import os
os.system("pip install ogb")
from sklearn.model_selection import train_test_split
import datetime
from model import SAGE_w_feat, SAGE_no_feat
import argparse
import random
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid

from logger import Logger
import pandas as pd

import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import pickle

import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer


forward_sample_size = None
super_loss = None
metric_sort_train = {}
metric_sort_val = {}

metric_to_consider = []

ling_features_traditional = ['FleschG_S', 'AutoRea_S', 'ColeLia_S', 'SmogInd_S', 'Gunning_S', 'LinseaW_S', 'TokSenM_S', 'TokSenS_S', 'TokSenL_S', 'as_Token_C', 'as_Sylla_C', 'at_Sylla_C', 'as_Chara_C', 'at_Chara_C']
ling_featuers_all = ['to_NoTag_C', 'as_NoTag_C', 'at_NoTag_C', 'ra_NoAjT_C', 'ra_NoVeT_C', 'ra_NoAvT_C', 'ra_NoSuT_C', 'ra_NoCoT_C', 'to_VeTag_C', 'as_VeTag_C', 'at_VeTag_C', 'ra_VeAjT_C', 'ra_VeNoT_C', 'ra_VeAvT_C', 'ra_VeSuT_C', 'ra_VeCoT_C', 'to_AjTag_C', 'as_AjTag_C', 'at_AjTag_C', 'ra_AjNoT_C', 'ra_AjVeT_C', 'ra_AjAvT_C', 'ra_AjSuT_C', 'ra_AjCoT_C', 'to_AvTag_C', 'as_AvTag_C', 'at_AvTag_C', 'ra_AvAjT_C', 'ra_AvNoT_C', 'ra_AvVeT_C', 'ra_AvSuT_C', 'ra_AvCoT_C', 'to_SuTag_C', 'as_SuTag_C', 'at_SuTag_C', 'ra_SuAjT_C', 'ra_SuNoT_C', 'ra_SuVeT_C', 'ra_SuAvT_C', 'ra_SuCoT_C', 'to_CoTag_C', 'as_CoTag_C', 'at_CoTag_C', 'ra_CoAjT_C', 'ra_CoNoT_C', 'ra_CoVeT_C', 'ra_CoAvT_C', 'ra_CoSuT_C', 'to_ContW_C', 'as_ContW_C', 'at_ContW_C', 'to_FuncW_C', 'as_FuncW_C', 'at_FuncW_C', 'ra_CoFuW_C', 'FleschG_S', 'AutoRea_S', 'ColeLia_S', 'SmogInd_S', 'Gunning_S', 'LinseaW_S', 'TokSenM_S', 'TokSenS_S', 'TokSenL_S', 'as_Token_C', 'as_Sylla_C', 'at_Sylla_C', 'as_Chara_C', 'at_Chara_C', 'SimpTTR_S', 'CorrTTR_S', 'BiLoTTR_S', 'UberTTR_S', 'MTLDTTR_S', 'SimpNoV_S', 'SquaNoV_S', 'CorrNoV_S', 'SimpVeV_S', 'SquaVeV_S', 'CorrVeV_S', 'SimpAjV_S', 'SquaAjV_S', 'CorrAjV_S', 'SimpAvV_S', 'SquaAvV_S', 'CorrAvV_S']

copy_of_current_batch_metric = {}
copy_precomputed_order = {}

df_train_metric = None
df_val_metric  = None
rbf_dict = {}

batch_loss_history = {}
global_batch_counter = 0
loss_history = {}
trend_history = {}
conf_history = {}
tau_history = {}
tau_adjusted_history = {}
loss_trend_conf_history = {}

def load_precomputed_metric_order(args):
    global copy_precomputed_order
    if args.transfer!= "none":
        values = args.transfer.split("_")
        source_model_type = values[0]
        source_feat = values[1]
        source_dataset = values[2]
        file_name = f'../logs/copy_of_current_batch_metric_{source_dataset}_{source_model_type}_{source_feat}.pkl'
        copy_precomputed_order = pickle.load(open(file_name, "rb"))


def set_metric_to_consider(args):
    global metric_to_consider
    if args.dataset == "arxiv" or "arxiv" in args.transfer:
        if args.use_nlp_indices:
            '''
            metric_to_consider = ["degree", "deg_cent", "density", "len_local_bridges", "add_avg_neighbor_deg",
                              "large_clique_size", "add_eigenvector_centrality_numpy", "degree_assortativity_coefficient", "ramsey_R2", "mean_degree_mixing_matrix"] + ling_features_traditional
            '''
            metric_to_consider = ling_features_traditional
            
        else:
            
            metric_to_consider = ["degree", "deg_cent", "density", "len_local_bridges", "add_avg_neighbor_deg",
                              "large_clique_size", "add_eigenvector_centrality_numpy", "degree_assortativity_coefficient", "ramsey_R2", "mean_degree_mixing_matrix"]
    
    elif args.dataset == "citeseer"  or "citeseer" in args.transfer:
        
        if args.dataset == "cora":
            metric_to_consider = [ 'degree','len_min_weighted_dominating_set','add_eigenvector_centrality_numpy', 'degree_assortativity_coefficient','avg_clustering','node_connectivity','add_average_degree_connectivity']
            
        else:
            
            metric_to_consider = [ 'degree','average_node_connectivity','len_min_weighted_dominating_set','add_eigenvector_centrality_numpy', 'degree_assortativity_coefficient','avg_clustering','len_local_bridges','density','node_connectivity','add_average_degree_connectivity']
        
        
    elif args.dataset == "cora"  or "cora" in args.transfer:
        metric_to_consider = ['treewidth_min_degree', 'node_connectivity', 'len_min_weighted_dominating_set','add_eigenvector_centrality_numpy','degree_assortativity_coefficient','degree', 'add_closeness_centrality','add_avg_neighbor_deg', 'avg_clustering', 'add_average_degree_connectivity' ]


def sort_metric_dataset(df_train_metric,args): # this also supports CCL from our spaced repetation paper

    metric_sort = {}
    if args.sum_of_all_metric:
        soam_dict = {}
        metric_sort["sum_of_all_metric"] = df_train_metric.sort_values(by="sum_of_all_metric",ascending=True).index.tolist()
        metric_sort.update(soam_dict)
        return metric_sort
    
    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            metric_sort[m + "_A"] = df_train_metric.sort_values(by=m,ascending=True).index.tolist()
        
        if "D" in args.metric_order:
            metric_sort[m + "_D"] = df_train_metric.sort_values(by=m,ascending=False).index.tolist()
        
        if "Q" in args.metric_order:
            complexity_scores = df_train_metric[m].to_numpy()
            data = QuantileTransformer(n_quantiles=len(complexity_scores), output_distribution='normal', random_state=0).fit_transform(complexity_scores.reshape(-1, 1))
            data = np.absolute(data - data.mean())
            #pdf = norm.pdf(data)  # probability of transformed samples
            data = normalize(data, axis=0) # l2 norm
            tmp = []
            for (k,i, j) in (sorted(zip(range(len(complexity_scores)), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=True)):
                tmp.append(k)
            metric_sort[m + "_Qd"] = tmp
            
            tmp = []
            for (k,i, j) in (sorted(zip(range(len(complexity_scores)), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=False)):
                tmp.append(k)
            metric_sort[m + "_Qa"] = tmp
        

    
    if args.add_random:
        if "A" in args.metric_order:
            metric_sort["random_A"] = df_train_metric.sort_values(by="random",ascending=True).index.tolist()
        if "D" in args.metric_order:
            
            metric_sort["random_D"] = df_train_metric.sort_values(by="random",ascending=False).index.tolist()
        if "Q" in args.metric_order:
            complexity_scores = df_train_metric["random"].to_numpy()
            data = QuantileTransformer(n_quantiles=len(complexity_scores), output_distribution='normal', random_state=0).fit_transform(complexity_scores.reshape(-1, 1))
            data = np.absolute(data - data.mean())
            data = normalize(data, axis=0) # l2 norm
            #pdf = norm.pdf(data)  # probability of transformed samples
            tmp = []
            for (k,i, j) in (sorted(zip(range(n_samples), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=True)):
                tmp.append(k)
            metric_sort["random_Qd"] = tmp
            
            tmp = []
            for (k,i, j) in (sorted(zip(range(n_samples), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=False)):
                tmp.append(k)
            metric_sort["random_Qa"] = tmp
            
    return metric_sort


def fix_negative_value_if_any(values):
    tmp = np.array(values)
    min_value = tmp.min()
    if min_value < 0:
        print("fixing negative values")
        tmp = tmp + (-1*min_value)
    return tmp


def load_train_metric(args):
    if args.dataset == "arxiv":
        if args.use_nlp_indices:
            df_train_metric = pd.read_csv("../indices/ogbn_arxiv_col_indices_w_nlp.csv")
        else:
            df_train_metric = pd.read_csv("../indices/ogbn_arxiv/ogbn_arxiv_col_indices.csv")
    if args.dataset == "citeseer":
        df_train_metric = pd.read_csv("../indices/citeseer/citeseer_col_indices_part_train.csv")
    if args.dataset == "cora":
        df_train_metric = pd.read_csv("../indices/cora/cora_col_indices_v2_80_10_10.csv")
    
        
    df_train_metric = df_train_metric.fillna(0)
    print("using l2 norm")
    for c in df_train_metric.columns.tolist()[2:]:
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c][:,np.newaxis], axis=0)
            
    if args.add_random:
        print("random added")
        df_train_metric['random'] = np.random.rand(len(df_train_metric)).tolist()
        df_train_metric['random'] = normalize(df_train_metric['random'][:,np.newaxis], axis=0)
        
    if args.sum_of_all_metric:
        print("performing sum of all metrics")
        metric_name_columns = df_train_metric.columns.tolist()[2:] 
        df_train_metric['sum_of_all_metric'] = df_train_metric[metric_name_columns].sum(axis=1)
 
        
    return df_train_metric


def load_val_metric(args):
    
    if args.dataset == "arxiv":
        if args.use_nlp_indices:
            df_val_metric = pd.read_csv("../indices/ogbn_arxiv_col_indices_w_nlp.csv")
        else:
            df_val_metric = pd.read_csv("../indices/ogbn_arxiv_col_indices_val.csv")
    if args.dataset == "citeseer":
        df_val_metric = pd.read_csv("../indices/citeseer_col_indices_part_val.csv")
    if args.dataset == "cora":
        df_val_metric = pd.read_csv("../indices/cora_col_indices_val.csv")
        
    df_val_metric = df_val_metric.fillna(0)
    print("using l2 norm")
    for c in df_val_metric.columns.tolist()[2:]:
            tmp = fix_negative_value_if_any(df_val_metric[c])
            df_val_metric[c] = tmp 
            df_val_metric[c] = normalize(df_val_metric[c][:,np.newaxis], axis=0)
            
    if args.add_random:
        print("random added")
        df_val_metric['random'] = np.random.rand(len(df_val_metric)).tolist()
        df_val_metric['random'] = normalize(df_val_metric['random'][:,np.newaxis], axis=0)
    
    if args.sum_of_all_metric: # this is used to compute the CCL baseline reported in the paper
        print("performing sum of all metrics")
        metric_name_columns = df_val_metric.columns.tolist()[2:] 
        df_val_metric['sum_of_all_metric'] = df_val_metric[metric_name_columns].sum(axis=1)
        
    return df_val_metric


def get_updated_idx(model, data, train_idx,args,c, epoch):
    global df_val_metric, df_train_metric
    global metric_sort_val, metric_sort_train
    
    if df_val_metric is None:
        df_val_metric = load_val_metric(args)
    
    if df_train_metric is None:
        df_train_metric = load_train_metric(args)
        
    if len(metric_sort_val) == 0:
        metric_sort_val = sort_metric_dataset(df_val_metric,args)
        
    if len(metric_sort_train) == 0:
        metric_sort_train = sort_metric_dataset(df_train_metric,args)   
        
    metric_name = []
    metric_loss_value = []
    nb_example_v = int(c * len(df_val_metric))
    
    model.eval()
    metric_sort = metric_sort_val if args.error_split == "val" else metric_sort_train
    for m in metric_sort:
        order = metric_sort[m]
        c_idx = order[:nb_example_v]
        
        out = model(data.x, data.adj_t)[c_idx]
        loss = F.nll_loss(out, data.y.squeeze(1)[c_idx], reduction = "mean")
        metric_name.append(m)
        metric_loss_value.append(loss.detach().cpu().item())

    metric_loss_value = np.array(metric_loss_value)
    j_idx = np.argmax(metric_loss_value)
    j = metric_name[j_idx]

        
    print(epoch, j)
    nb_example_t = int(c * len(train_idx))
    updated_train_idx = metric_sort_train[j][:nb_example_t]
    updated_train_idx = torch.tensor(updated_train_idx)
    return updated_train_idx


def get_combined_idx (args,c, rbf_current_batch):
    
    global df_val_metric, df_train_metric
    global metric_sort_val, metric_sort_train
    
    if df_val_metric is None:
        df_val_metric = load_val_metric(args)
    
    if df_train_metric is None:
        df_train_metric = load_train_metric(args)
        
    if len(metric_sort_val) == 0:
        metric_sort_val = sort_metric_dataset(df_val_metric,args)
        
    if len(metric_sort_train) == 0:
        metric_sort_train = sort_metric_dataset(df_train_metric,args)   
        
    combined_idx = []
    nb_example_v = int(c * len(df_val_metric))

    metric_sort = metric_sort_train
    for m in metric_sort:
        if m in rbf_current_batch:
            order = metric_sort[m]
            c_idx = order[:nb_example_v]
            combined_idx.extend(c_idx)

    combined_idx = np.array(list(set(combined_idx)))
    combined_idx = torch.tensor(combined_idx)
    return combined_idx.long()



def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])


def update_competency_equal_dist(t, T, c_0, a):
    if T == t:
        return 1
    else:
        term = pow(1-((1-c_0)*(1-(t/T))), a)
        return min([1,term])   

def train(model, data, train_idx, optimizer, args, epoch):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]

    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx], reduction = "none")
    loss = loss.mean()
    #loss = calculate_superloss(loss, epoch, train_idx, args)

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def init_rbf_dict(args):
    global df_val_metric, df_train_metric
    global metric_sort_val, metric_sort_train
    
    if df_val_metric is None:
        df_val_metric = load_val_metric(args)
    
    if df_train_metric is None:
        df_train_metric = load_train_metric(args)
        
    if len(metric_sort_val) == 0:
        metric_sort_val = sort_metric_dataset(df_val_metric,args)
        
    if len(metric_sort_train) == 0:
        metric_sort_train = sort_metric_dataset(df_train_metric,args)   
    
    global rbf_dict
    
    for m in metric_sort_train:
        rbf_dict[m] = 1

def reduce_one_in_delayed_batch(rbf_delayed_batch):
    global rbf_dict
    for m in rbf_dict:
        if m in rbf_delayed_batch:
            rbf_dict[m] = rbf_dict[m] - 1 


def get_optimal_tau_rbf(kern, x, proba, losses, nu):

        if kern == 'gau':

            a_ln = -1. * np.sum([np.log(a) for a in proba if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(losses, proba) if a >= nu])
            tau = a_ln / x_sum_pow

        if kern == 'lap':
            a_ln = -1. * np.sum([np.log(a) for a in proba if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(losses, proba) if a >= nu])                        
            tau = a_ln / x_sum
            
        
        if kern == 'lin':
            a_one = np.sum([(1. - a) for a in proba if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(losses, proba) if a >= nu])                            
            tau = a_one / x_sum
        
        if kern == 'cos':
            a_arc = np.sum([np.arccos(2. * a - 1.) for a in proba if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(losses, proba) if a >= nu])
            tau = a_arc / (np.pi * x_sum)
        
        if kern == 'qua':
            a_one = np.sum([(1. - a) for a in proba if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(losses, proba) if a >= nu])           
            tau = a_one / x_sum_pow
        
        if kern == 'sec':
            a_sq = np.sum([np.log(1. / a + np.sqrt(1. / a - 1.)) for a in proba if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(losses, proba) if a >= nu])                            
            tau = a_sq / x_sum

        return tau 


def calculate_forward_pass_rbf(model, c, metric_name, val_idx, data, evaluator):
    model.eval()
    nb_example_t = int(c * len(val_idx))
    print("nb_example_t", nb_example_t)
    c_idx = metric_sort_val[metric_name][:nb_example_t]
    c_idx = random.sample(c_idx, forward_sample_size) if len(c_idx) > forward_sample_size else c_idx
    out = model(data.x, data.adj_t)[c_idx]
    loss_vec = F.nll_loss(out, data.y.squeeze(1)[c_idx], reduction = "none").detach().cpu().numpy()
    loss_avg = F.nll_loss(out, data.y.squeeze(1)[c_idx], reduction = "mean").detach().cpu().numpy()
    #print("\t metric_name = {} loss_vec = {}".format(metric_name, loss_vec))
    #print("\t metric_name = {} loss_avg = {}".format(metric_name, loss_avg))
    m = nn.Softmax(dim=1)
    out_softmax = m(out)
    proba, prediction_class = torch.max(out_softmax, dim=1)
    proba = proba.detach().cpu().numpy()
    prediction_class = prediction_class.detach().cpu().numpy()
    
    gt = data.y.squeeze(1)[c_idx].detach().cpu().numpy()

    performance = accuracy_score(gt, prediction_class)
    
    return proba, loss_avg, loss_vec,performance

def get_delay(kern, tau_hat, s_e, metric_loss, nu):
    t_hat = None
    vec = None
    
    if kern == "gau":
        if s_e >= nu:

            nu_gau = np.sqrt(-np.log(nu) / tau_hat)
            vec = s_e*nu_gau/metric_loss
            t_hat = np.mean(vec)
    
    if kern == 'lap':
        if s_e >= nu:
            nu_lap = np.log(nu)
            vec = -1. * s_e * nu_lap / (metric_loss * tau_hat)
            t_hat = np.mean(vec)
                        
    if kern == 'lin':
        nu_lin = (1. - nu)
        if s_e >= nu:
            vec = 1. * s_e * nu_lin / (metric_loss * tau_hat)
            t_hat = np.mean(vec)
                        
    if kern == 'cos':
        nu_cos = np.arccos(2 * nu - 1.)
        if s_e >= nu:
            vec = s_e * nu_cos / (np.pi * metric_loss * tau_hat)
            t_hat = np.mean(vec)
                        
    if kern == 'qua':
        nu_qua = np.sqrt((1. - nu) / tau_hat)
        if s_e >= nu:
            vec = s_e * nu_qua / metric_loss
            t_hat = np.mean(vec)
                            
    if kern == 'sec':
        nu_sec = np.log(1. / nu * (1 + np.sqrt(1 - nu * nu)))
        if s_e >= nu:
            vec = s_e * nu_sec / (metric_loss * tau_hat)
            t_hat = np.mean(vec)

    if not vec is None:
        
        print("\t t_hat vec: ",vec.min(), vec.max(), vec.std(), vec.mean(), np.isnan(vec), np.isinf(vec))
        
    else:
        print("\t vec: ", vec)
    print("\t s_e > = nu", s_e >=nu)
    print("\t t_hat: ", t_hat)

    if t_hat is None or np.isnan(t_hat) or np.isinf(t_hat):
        t_hat = 1
    print("\t t_hat (nan/inf):", np.isnan(t_hat), np.isinf(t_hat))
    return t_hat

def calculate_delay_for_current_batch(args,eta, model, c, rbf_current_batch, val_idx, data, evaluator, epoch):
    # for tau_hat 
    global  rbf_dict
    for metric_name in rbf_dict:
        
        kern = args.rbf
        nu =eta
        proba, loss_avg, loss_vec,performance = calculate_forward_pass_rbf(model, c, metric_name, val_idx, data, evaluator)    
        x = 1.0/performance
        if metric_name in rbf_current_batch:
            tau_hat = get_optimal_tau_rbf(kern, x, proba, loss_vec, nu)
            #tau_hat = get_optimal_tau_rbf_wo_nu(kern, x, proba, loss_vec, nu)
            t_hat = get_delay(kern, tau_hat, performance, loss_vec, nu)
        else:
            tau_hat = "none"
            t_hat = rbf_dict[metric_name]
            
        if metric_name in rbf_current_batch:
            rbf_dict[metric_name] = t_hat
        
        else:
            pass
        print("\t epoch = {} \t tau_hat = {} \t t_hat = {} \t s_e = {:.4f} \t loss_avg = {:.4f} \t metric_name = {}".format(epoch, tau_hat, t_hat, performance,loss_avg,  metric_name))


def display_rbf_dict(rbf_dict, e):
    for m in rbf_dict:
        print("\t\t\t rbf_dict at epoch = {} \t {:.4f} \t {}".format(e, rbf_dict[m], m))



def get_arxiv_dataset(root):
    dataset = PygNodePropPredDataset('ogbn-arxiv', root=root,
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    #data = data.to(device)

    split_idx = dataset.get_idx_split()
    #train_idx = split_idx['train'].to(device)
    return split_idx, data, dataset


def get_citeseer_dataset():
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', split="full")
    data = dataset[0]
    data.y = data.y.reshape(-1, 1)
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()
    #data = data.to(device)
    split_idx = {}
    split_idx["train"] = torch.nonzero(data.train_mask).reshape(-1)
    split_idx["valid"] = torch.nonzero(data.val_mask).reshape(-1)
    split_idx["test"] = torch.nonzero(data.test_mask).reshape(-1)
    #train_idx = split_idx['train'].to(device)

    return split_idx, data, dataset


def get_cora_dataset():
    dataset = Planetoid(root='/tmp/Cora', name='Cora', split="full", num_train_per_class=380, num_val=271,
                        num_test=271)
    data = dataset[0]
    data.y = data.y.reshape(-1, 1)
    data = T.ToSparseTensor()(data)
    data.adj_t = data.adj_t.to_symmetric()

    # creating a new split in cora, which overwrites default split
    nodes = data.train_mask.size()[0]
    nodes = range(nodes)

    nodes_train, nodes_test, y_train, y_test = train_test_split(nodes, data.y.tolist(), stratify=data.y.tolist(),
                                                                test_size=0.2, random_state=42)
    nodes_val, nodes_test, y_val, y_test = train_test_split(nodes_test, y_test, stratify=y_test, test_size=0.5,
                                                            random_state=42)

    data.train_mask = torch.tensor([True if i in nodes_train else False for i in nodes]).bool()
    data.val_mask = torch.tensor([True if i in nodes_val else False for i in nodes]).bool()
    data.test_mask = torch.tensor([True if i in nodes_test else False for i in nodes]).bool()

    print(data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum())

    # As plantoid is different from arxiv, it does not have split_idx by default
    # so creating split_idx explicitly
    #data = data.to(device)
    split_idx = {}
    split_idx["train"] = torch.nonzero(data.train_mask).reshape(-1)
    split_idx["valid"] = torch.nonzero(data.val_mask).reshape(-1)
    split_idx["test"] = torch.nonzero(data.test_mask).reshape(-1)
    #train_idx = split_idx['train'].to(device)

    return split_idx, data, dataset


def run(args):
    global forward_sample_size
    c_0 = 0.01

    #delay_matrices_log = {}
    load_precomputed_metric_order(args)
    set_metric_to_consider(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset == "arxiv":
        root = "/tmp/datasets"
        split_idx,data,dataset = get_arxiv_dataset(root)
        data = data.to(device)
        
    if args.dataset == "citeseer":
        split_idx,data,dataset = get_citeseer_dataset()
        data = data.to(device)
        
    if args.dataset == "cora":
        split_idx,data,dataset = get_cora_dataset()
        data = data.to(device)

    forward_sample_size = int(len(split_idx["valid"]) * 0.01)

    if args.feat == "yes": # args.feat = yes indicates GTNN as base model
        model = SAGE_w_feat(data.num_features, args.hidden_channels,
                         dataset.num_classes, args.num_layers,
                         args.dropout, data.x, args.feat).to(device)
    else:
        model = SAGE_no_feat(data.num_features, args.hidden_channels,
                         dataset.num_classes, args.num_layers,
                         args.dropout, data.x, args.model_type).to(device)   

    evaluator = Evaluator(name='ogbn-arxiv')
    
    logger = Logger(args.runs, args)
    #sys.stdout = sys.stderr = open(logger.get_log_file(args), "w")
    print(args)


    #==========================================================
    # Training starts
    # ==========================================================


    for run in range(args.runs):
        init_rbf_dict(args)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.l2)
        for epoch in range(1, 1 + args.epochs):

            if args.rbf != "none": # args.rbf condition is used to support running current model and baselines
                if args.transfer != "none":                   
                    rbf_current_batch = copy_precomputed_order[epoch]
                    rbf_delayed_batch = []
                else:
                    display_rbf_dict(rbf_dict, epoch)
                    rbf_current_batch = [m for m in rbf_dict if rbf_dict[m] <= 1]
                    rbf_delayed_batch = [m for m in rbf_dict if rbf_dict[m] > 1]
                    print("\t rbf_current_batch: ",rbf_current_batch)
                    print("\t rbf_delayed_batch: ",rbf_delayed_batch)
                    copy_of_current_batch_metric[epoch] = rbf_current_batch

            c = c_0 if epoch == 1 else c
            train_idx = split_idx['train'].to(device)
            
            if args.rbf != "none":
                train_idx = get_combined_idx(args,c,rbf_current_batch)
            else:
                train_idx = get_updated_idx(model, data, train_idx, args,c, epoch) if args.approach == "loss_based" else train_idx

            model.train()
            print("total number of examples: ", len(train_idx))
            loss = train(model, data, train_idx, optimizer, args,epoch)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

            # ==========================================================
            # Update Competence value as per the desired function
            # ==========================================================
            sys.stdout.flush()
            if args.competence == "plato":
                c = update_competency(epoch, args.epochs, c_0, 2)
            elif args.competence == "equal_dist":
                c = update_competency_equal_dist(epoch, args.epochs, c_0, args.a)
                print("updated c: ", c)
            else:
                c = None
         
            if args.rbf != "none":
                val_idx = split_idx['valid']
                model.eval()

                '''
                uncomment the lines below to calculate the loss of every example on validation dataset.
                
                out = model(data.x, data.adj_t)
                m = nn.Softmax(dim=1)
                out_softmax = m(out)[val_idx]
                proba, prediction_class = torch.max(out_softmax, dim=1)
                proba = proba.detach().cpu().numpy()
                prediction_class = prediction_class.detach().cpu().numpy()
                df_val_prediction_per_epoch[epoch] = prediction_class
                '''

                eta = args.eta
                print("\t updated value of eta: ", eta)

                if args.competence == "plato":
                    c_v = update_competency(epoch, args.epochs, c_0, 4)
                elif args.competence == "equal_dist":
                    c_v = update_competency_equal_dist(epoch, args.epochs, c_0, args.a)
                else:
                    c_v = None
                print("c_v", c_v)
                calculate_delay_for_current_batch(args,eta, model, c_v, rbf_current_batch, val_idx, data,evaluator, epoch)
                reduce_one_in_delayed_batch(rbf_delayed_batch)
            sys.stdout.flush()

                
        logger.print_statistics(run)
        out = model(data.x, data.adj_t)

        '''
        uncomment this to calculate the loss of every example on validation dataset.
        df_val_prediction_per_epoch["ground_truth"] = data.y[val_idx].detach().cpu()
        '''

        '''
        uncomment the line below to calculate the loss of every example on validation dataset.
        df_val_prediction_per_epoch["ground_truth"] = data.y[val_idx].detach().cpu()
        '''

        '''
        uncomment the lines below to save predictions with prob values for future use.
        #np.savetxt('node_prediction/train_arxiv_no.out', out[split_idx['train']].detach().cpu().numpy(), delimiter=',') 
        #np.savetxt('node_prediction/val_arxiv_no.out', out[split_idx['valid']].detach().cpu().numpy(), delimiter=',') 
        #np.savetxt('node_prediction/test_arxiv_no.out', out[split_idx['test']].detach().cpu().numpy(), delimiter=',') 
        '''


    logger.print_statistics()

    
    '''
    uncomment this to calculate the loss of every example on validation dataset.
    df_val_prediction_per_epoch.to_csv("../logs/val_prediction_per_epoch.csv")
    '''   


if __name__ == "__main__":

    '''
    for rbf, set following arguments:
        training_type = regular
        rbf = gau/lap/lin/cos/sec/qua
        eta = choose between 0-1
        competence = equal_dist
        metric_order = ADQaQd
        approach = none
        
    for CCL, set the following arguments:
        training_type = regular
        rbf = none
        eta = 0
        competence = plato
        sum_of_all_metric = True # default value is False
        metric_order = A
        approach = loss_based
        
    for No-CL, set the following arguments:
        training_type = regular
        rbf = none
        eta = 0
        competence = none
        approach = none
        
    '''

    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    #parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--feat', type=str, default="yes")

    parser.add_argument('--dataset', type=str, default="arxiv")

    parser.add_argument('--model_type', type=str, default="sage")


    parser.add_argument('--training_type', type=str, default="regular",
                        help="type of training as regular")

    # metric order and loss creteria
    parser.add_argument('--approach', type=str, default="none", help="none, loss_based")
    #parser.add_argument('--loss_creteria', type=str, default='max', help="loss_creteria: min/max")
    parser.add_argument('--metric_order', type=str, default="ADQaQd",
                        help=" choose for ascending order:A/D/Qa/Qd/ADQaQd")
    parser.add_argument('--add_random', type=bool, default=False, help="T/F")
    parser.add_argument('--use_k_means', type=bool, default=True, help="T/F")
    parser.add_argument('--error_split', type=str, default="val", help="train/val")

    # rbf
    parser.add_argument('--rbf', type=str, default="none", help="none/gau/lap/lin/cos/sec/qua")
    parser.add_argument('--eta', type=float, default=0.5, help="choose between 0-1")

    # for competence update
    parser.add_argument('--competence', type=str, default="equal_dist", help="plato/equal_dist")
    parser.add_argument('-a', type=float, default=0.5, help="choose between 0-1")
    parser.add_argument('--use_nlp_indices', type=bool, default=False, help="T/F")
    parser.add_argument('--transfer', type=str, default="none", help="sage_yes_arxiv")
    # for CCL baseline reported in the paper
    parser.add_argument('--sum_of_all_metric', type=bool, default=False, help="T/F")

    args = parser.parse_args()

    start_time = datetime.datetime.now()
    run(args)
    end_time = datetime.datetime.now()

    diff = end_time - start_time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    print(f'hours = {hours}, minutes = {minutes}, seconds = {seconds}')
    sys.stdout.flush()
