import utils, models, g_metric, g_metric_val
from torch_geometric.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from sklearn.utils.class_weight import compute_class_weight
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

bce_loss = nn.BCEWithLogitsLoss(reduction='none')
cat_loss = nn.CrossEntropyLoss(reduction='none')
optimizer = None
forward_pass_loss = {} 
forward_key2prediction = {}
forward_labels = {}
key2idx = {}
forward_proba = {}

forward_sample_size  = None
node_feature_matrix = None


df_train_metric = None


def init_model(args):
    global node_feature_matrix
    device = args.device
    node_feature_dim = 300
    gs_dim = 100
    use_additional_features = args.add_additional_feature
    
    if args.dataset == "pgr":
        node_feature_dim = 300
        gs_dim = 100
        add_additional_feature_dim = 768
        nb_classes = 1

    if args.dataset == "omim":
        node_feature_dim = 300
        gs_dim = 100
        add_additional_feature_dim = 4
        nb_classes=1

    model = models.GTNN_outer(node_feature_dim, device, gs_dim, add_additional_feature_dim, nb_classes,use_additional_features, args.model_type)

    return model

def get_weighted_loss(loss, label, device):
    classes = [l.item() for l in label]

    if len(classes) > sum(classes) > 0 :
        cls_weight = compute_class_weight("balanced", [0,1], classes)
        cls_weight = cls_weight.tolist()
        weight = torch.tensor(cls_weight).to(device)
        weight_ = weight[label.data.view(-1).long()].view_as(label)
        loss = loss * weight_
    return loss


def get_key2idx(train_set):
    global key2idx
    for idx,t in enumerate(train_set):
        t.key[-1] = str(t.key[-1])
        t.key = [str(k) for k in t.key]
        key2idx[" ".join(t.key)] = idx

def seperate_seen_from_new_edges(data_loader,train_set):
    edge_to_be_processed = []
    pre_computed_loss_values = []
    predictions = []
    labels = []
    proba = []
    for batch in data_loader:
        keys = batch.key
        for k in keys:
            k = [str(j) for j in k]
            k[-1] = str(k[-1])
            str_k = " ".join(k)
            if forward_pass_loss.get(str_k) is None:
                edge = train_set[key2idx[str_k]]
                edge_to_be_processed.append(edge)
            else:
                pre_computed_loss_values.append(forward_pass_loss.get(str_k))
                predictions.append(forward_key2prediction[str_k])
                labels.append(forward_labels[str_k])
                proba.append(forward_proba[str_k])
    
    return pre_computed_loss_values, edge_to_be_processed, predictions, labels, proba

#@profile
def calculate_error_metric(model, data_loader, device, train_set,args):
    global forward_pass_loss
    global forward_key2prediction
    global forward_labels
    global forward_proba

    saved_computed_loss_values, edge_to_be_processed, saved_predictions, saved_labels, saved_proba = seperate_seen_from_new_edges(
        data_loader, train_set)
    # print(len(edge_to_be_processed))
    updated_dataloader = DataLoader(edge_to_be_processed, batch_size=32,
                                    shuffle=False, pin_memory=True, num_workers=0)

    model.eval()
    batch_loss = []
    batch_predictions = []
    batch_labels = []
    batch_proba = []

    with torch.no_grad():
        counter = 0
        for batch in updated_dataloader:
            
            batch = batch.to(device)
            label = batch.y
            
            prediction = model(batch) # omim / pgr -- logits

            prediction_proba = F.sigmoid(prediction)
            batch_predictions.extend(prediction_proba.cpu().numpy().tolist())
            batch_proba.extend(prediction_proba.cpu().numpy().tolist())

            loss = bce_loss(prediction_proba.float(), label.float())
            loss = loss.cpu().detach()

            batch_loss.extend(loss.cpu().numpy().tolist())
            batch_labels.extend(label.cpu().numpy().tolist())

            for key, l, p, lb, pb in zip(batch.key, loss, prediction_proba, label, prediction_proba):
                key[-1] = str(key[-1])
                str_k = " ".join(key)
                forward_pass_loss[str_k] = l
                forward_key2prediction[str_k] = p
                forward_labels[str_k] = lb
                forward_proba[str_k] = pb

    if len(edge_to_be_processed) > 0:
        if len(saved_labels) > 0:
            combined_loss = batch_loss + [i.item() for i in saved_computed_loss_values]
            combined_predictions = batch_predictions + [i.item() for i in saved_predictions]
            combined_proba = batch_proba + [i.item() for i in saved_proba]
            combined_labels = batch_labels + [i.item() for i in saved_labels]
        else:
            combined_loss = batch_loss
            combined_predictions = batch_predictions
            combined_proba = batch_proba
            combined_labels = batch_labels
    else:
        combined_loss = [i.item() for i in saved_computed_loss_values]
        combined_predictions = [i.item() for i in saved_predictions]
        combined_labels = [i.item() for i in saved_labels]
        combined_proba = [i.item() for i in saved_proba]
        
    combined_loss = np.array(combined_loss)
    combined_predictions = np.array(combined_predictions)
    combined_labels = np.array(combined_labels)
    combined_proba = np.array(combined_proba)

    if args.rbf != 'none':
        combined_predictions[combined_predictions >= 0.5] = 1
        combined_predictions[combined_predictions < 0.5] = 0
        performance = f1_score(combined_labels, combined_predictions)
    else:
        performance = None

    return combined_loss.mean(), combined_loss, performance, combined_proba



def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])

def update_competency_equal_dist(t, T, c_0, a):
    if T == t:
        return 1
    else:
        term = pow(1-((1-c_0)*(1-(t/T))), a)
        return min([1,term])   

def calculate_error_metric_using_complexity_score(metric, c, train_set):
    if metric.endswith("_A"):
        metric = metric.replace("_A", "")
        D = df_train_metric.sort_values(by=metric,ascending=True).index.tolist()
    if metric.endswith("_D"):
        metric = metric.replace("_D", "")
        D = df_train_metric.sort_values(by=metric,ascending=False).index.tolist()
    if metric.endswith("_Qa") or metric.endswith("_Qd"):
        metric = metric.replace("_Qa", "").replace("_Qd", "")
        complexity_scores = df_train_metric[metric].to_numpy()
        data = QuantileTransformer(n_quantiles=len(complexity_scores), output_distribution='normal', random_state=0).fit_transform(complexity_scores.reshape(-1, 1))
        data = np.absolute(data - data.mean())
        #pdf = norm.pdf(data)  # probability of transformed samples
        tmp = []
        reverse = True if metric.endswith("_Qd") else False
        for (k,i, j) in (sorted(zip(range(len(complexity_scores)), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=reverse)):
             tmp.append(k)
        D = tmp
        
    nb_examples = int(c * len(train_set))
    D_trim = D[:nb_examples]
    return df_train_metric[metric][D_trim].mean()

def fix_negative_value_if_any(values):
    tmp = np.array(values)
    min_value = tmp.min()
    if min_value < 0:
        print("fixing negative values")
        tmp = tmp + (-1*min_value)
    return tmp

def read_train_metric(args):
    
    if args.dataset == "pgr":
        df_train_metric = pd.read_csv("../indices/pgr_col_indices_part5.csv")
        print("using l2 norm")
        for c in df_train_metric.columns.tolist()[4:]:
            
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c][:,np.newaxis], axis=0)
            
    elif args.dataset == "omim":
        df_train_metric = pd.read_csv("../indices/omim_col_indices_v2.csv")
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
        print("perfroming sum of all metrics")
        metric_name_columns = df_train_metric.columns.tolist()[4:] if args.dataset == "pgr" else df_train_metric.columns.tolist()[2:]
        df_train_metric['sum_of_all_metric'] = df_train_metric[metric_name_columns].sum(axis=1)

    return df_train_metric


def read_val_metric(args):
    
    if args.dataset == "pgr":
        df_val_metric = pd.read_csv("../indices/pgr_col_indices_part5_val.csv")
        print("using l2 norm")
        for c in df_val_metric.columns.tolist()[4:]:
            
            tmp = fix_negative_value_if_any(df_val_metric[c])
            df_val_metric[c] = tmp 
            df_val_metric[c] = normalize(df_val_metric[c][:,np.newaxis], axis=0)
            
    elif args.dataset == "omim":
        df_val_metric = pd.read_csv("../indices/omim_col_indices_val.csv")
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
        
    if args.sum_of_all_metric:
        print("perfroming sum of all metrics")
        metric_name_columns = df_val_metric.columns.tolist()[4:] if args.dataset == "pgr" else df_val_metric.columns.tolist()[2:]
        df_val_metric['sum_of_all_metric'] = df_val_metric[metric_name_columns].sum(axis=1)
 
    global forward_sample_size 
    forward_sample_size = int(len(df_val_metric) * 0.01)
    
    return df_val_metric

def get_optimal_tau_rbf(kern, x, proba, losses, nu):
        
        if kern == 'gau':
            a_ln = -1. * np.sum([np.log(a) for a in proba if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(losses, proba) if a >= nu])
            tau = a_ln / x_sum_pow
            #print("a_sq: ", a_sq)
            #print("x_sum_pow: ", x_sum_pow)
        
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
            #print("a_arc: ", a_arc)
            #print("x_sum: ", x_sum)
        
        if kern == 'qua':
            a_one = np.sum([(1. - a) for a in proba if a >= nu])
            x_sum_pow = np.sum([pow(l * x, 2) for l, a in zip(losses, proba) if a >= nu])           
            tau = a_one / x_sum_pow
        
        if kern == 'sec':
            a_sq = np.sum([np.log(1. / a + np.sqrt(1. / a - 1.)) for a in proba if a >= nu])
            x_sum = np.sum([l * x for l, a in zip(losses, proba) if a >= nu])                            
            tau = a_sq / x_sum
        #print("tau_hat: ", tau)
        return tau 

def get_delay(kern, tau_hat, s_e, metric_loss, nu):
    t_hat = 1
    
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
        
    if np.isnan(t_hat) or np.isinf(t_hat):
        t_hat = 1
    return t_hat

def display_rbf_dict(rbf_dict, e):
    for m in rbf_dict:
        print("\t {} \t {:.4f} \t {}".format(e, rbf_dict[m], m))


#@profile
def train_model_w_competence(train_set, eval_set,  args):
    eta = float(args.eta)
    rbf_dict = {}
    
    val_set = eval_set
    global df_train_metric
    df_train_metric = read_train_metric(args)
    df_val_metric = read_val_metric(args)
    global forward_pass_loss
    
    if args.error_split == 'val':
        get_key2idx(val_set)
    else:
        get_key2idx(train_set)
        
    approach = args.approach
    bs = args.batch_size
    c_0 = 0.1

    g_metric.sort_metric_dataset(df_train_metric, args)
    g_metric.revise_metric_dataloader(bs, train_set, c_0, args)
    metric_dict_train = g_metric.revise_metric_dict(args)

    if args.error_split == "val":
        g_metric_val.sort_metric_dataset_val(df_val_metric, args)
        g_metric_val.revise_metric_dataloader_val(bs, val_set, c_0, args, forward_sample_size)
        metric_dict_val = g_metric_val.revise_metric_dict_val(args)

    global optimizer

    lr = args.lr
    L2 = args.l2
    device = args.device
    model = init_model(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=L2)

    metric_dict = metric_dict_val if args.error_split == "val" else metric_dict_train
    
    if args.rbf != "none":
        for metric in metric_dict:
            rbf_dict[metric] = 1

    split_set = val_set if args.error_split == "val" else train_set

    metric_loss_values = []
    metrics = []
    for metric in metric_dict:
        #print(metric)
        if approach == "loss_based":
            metric_loss,metric_loss_vec, s_e, _ = calculate_error_metric(model, metric_dict[metric], device, split_set, args)

        metric_loss_values.append(metric_loss)
        metrics.append(metric)

    # train model on the revise dataset based on competency

    T = args.curriculum_length
    best_pref = -1.0

    for t in range(1, T):
        
        ("*"*80)
        if args.rbf != "none":
            #display_rbf_dict(rbf_dict, t)
            rbf_current_batch = [m for m in rbf_dict if rbf_dict[m] <= 1]
            rbf_delayed_batch = [m for m in rbf_dict if rbf_dict[m] > 1]
            if len(rbf_current_batch) == 0 :
                continue
            #print("\t rbf_current_batch: ",rbf_current_batch)
            #print("\t rbf_delayed_batch: ",rbf_delayed_batch)
            updated_metric_loss_values = []
            updated_metrics =[]
            for m, l in zip(metrics, metric_loss_values):
                if m in rbf_current_batch:
                    updated_metrics.append(m)
                    updated_metric_loss_values.append(l)
            metric_loss_values = updated_metric_loss_values
            metrics = updated_metrics
        
        forward_pass_loss = {}
        if args.loss_creteria == 'max':
                j = metrics[np.argmax(metric_loss_values)]
        else:
                j = metrics[np.argmin(metric_loss_values)]
        
        if args.rbf != "none":
            final_dataloader = g_metric.revise_metric_dataloader_combined(t, bs, train_set, c_0 if t == 1 else c, metrics)
        else:
            final_dataloader = metric_dict_train[j]
        model.train()
        losses = []
        for batch in final_dataloader:
                
            optimizer.zero_grad()
            batch = batch.to(device)
            label = batch.y

            prediction = model(batch)
            loss = bce_loss(prediction.float(), label.float())
            if args.dataset == "omim":
                loss = get_weighted_loss(loss, label, device)
                

            
            loss = loss.mean()
            losses.append(loss.item())
            if args.rbf == "none":
                print(t, j, loss.item())
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu()
            loss = None
            prediction = None
            batch = None
            del batch
            del loss
            del prediction
            torch.cuda.empty_cache()
            # break

        # save model on every competency
        
        eval_data_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=args.num_workers)

        p, r, f1, predictions, predictions_proba = evaluate.eval_model(model, eval_data_loader, device)
        eval_ds_name = "test" if args.evaluate_test_per_epoch else "val"
        print(
            "eval {0} t = {1:d} loss = {2:.6f} p = {3:.4f} r = {4:.4f} f1 = {5:.4f} ".format(eval_ds_name, t,
                                                                                             sum(losses) / len(losses),
                                                                                             p,
                                                                                             r, f1))
        current_pref = f1

        if current_pref > best_pref or t == 0:
            best_pref = current_pref
            utils.save_the_best_model(model, t, optimizer, {"p": p, "r": r, "f1": f1}, args)
        else:
            pass

        sys.stdout.flush()

        # update competency AND calculates the error for next bucket
        if args.competence == "plato":
            c = update_competency(t, T, c_0, 2)
        elif args.competence == "equal_dist":
            c = update_competency_equal_dist(t,T, c_0, args.a)
        else:
            c = None

        print(t, c)

        g_metric.revise_metric_dataloader(bs, train_set, c, args)
        metric_dict_train = g_metric.revise_metric_dict(args)

        if args.error_split == "val":
            g_metric_val.sort_metric_dataset_val(df_val_metric, args)
            g_metric_val.revise_metric_dataloader_val(bs, val_set, c, args, forward_sample_size)
            metric_dict_val = g_metric_val.revise_metric_dict_val(args)

        metric_dict = metric_dict_val if args.error_split == "val" else metric_dict_train
        split_set = val_set if args.error_split == "val" else train_set

        metric_loss_values = []
        metrics = []
        for metric in metric_dict:

            if approach == "loss_based":
                metric_loss, metric_loss_vec, s_e, proba_vec = calculate_error_metric(model, metric_dict[metric], device, split_set, args)
            metric_loss_values.append(metric_loss)
            metrics.append(metric)

            if args.rbf != "none":
                metric_loss_vec = np.array(metric_loss_vec)
                x = 1.0/s_e
                if metric in rbf_current_batch:
                    tau_hat = get_optimal_tau_rbf(args.rbf, x, proba_vec, metric_loss_vec, args.eta)
                    t_hat = get_delay(args.rbf, tau_hat, s_e, metric_loss_vec, args.eta) #metric_loss_vec
                else:
                    tau_hat = "none"
                    t_hat = rbf_dict[metric]
                #print("\t t_i = {:.4f} d_i = {:.4f} s_e = {:.4f} t_hat = {:.4f} tau_hat  = {} metric = {} ".format(t, metric_loss, s_e, t_hat, tau_hat, metric))
                if metric in rbf_current_batch:
                    rbf_dict[metric] = t_hat

        if args.rbf != "none":
            for m in rbf_delayed_batch:
                rbf_dict[m] = rbf_dict[m] - 1 #if rbf_dict[metric] > 1 else 1
        #display_rbf_dict(rbf_dict, t)
    sys.stdout.flush()
    return model


def train_model_gtnn(train_set, eval_set,  args):

    global optimizer
    lr = args.lr
    L2 = args.l2
    device = args.device
    model = init_model(args)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=L2)
    T = args.curriculum_length
    best_pref = -1.0
    train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=args.num_workers)
        
    for t in range(1, T):
        
        model.train()
        losses = []
        for batch in train_data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            label = batch.y

            prediction = model(batch)
            loss = bce_loss(prediction.float(), label.float())
            #loss = get_weighted_loss(loss, label, device)
            loss = loss.mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # to clean memory
            loss = loss.detach().cpu()
            loss = None
            prediction = None
            batch = None
            del batch
            del loss
            del prediction
            torch.cuda.empty_cache()
            # break

        # save model on every competency
        eval_data_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=args.num_workers)

        p, r, f1, _, _ = evaluate.eval_model(model, eval_data_loader, device)
        eval_ds_name = "test" if args.evaluate_test_per_epoch else "val"
        print(
            "eval {0} t = {1:d} loss = {2:.6f} p = {3:.4f} r = {4:.4f} f1 = {5:.4f} ".format(eval_ds_name, t,
                                                                                             sum(losses) / len(losses),
                                                                                             p,
                                                                                             r, f1))
        current_pref = f1

        if current_pref > best_pref or t == 0:
            best_pref = current_pref
            utils.save_the_best_model(model, t, optimizer, {"p": p, "r": r, "f1": f1}, args)
        else:
            pass

        sys.stdout.flush()

    return model
