from torch_geometric.data import DataLoader
import numpy as np
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import random

metric_to_consider = ["add_average_degree_connectivity",
                      "add_avg_neighbor_deg",
                      "add_eigenvector_centrality_numpy",
                      "add_katz_centrality_numpy",
                    "avg_clustering",
                    "common_neighbors",
                    "deg_cent",
                    "degree",
                    "degree_assortativity_coefficient",
                    "density",
                    "group_degree_centrality",
                    "large_clique_size",
                    "len_local_bridges",
                    "len_min_edge_dominating_set",
                    "len_min_maximal_matching",
                    "len_min_weighted_dominating_set",
                    "len_min_weighted_vertex_cover",
                    "ramsey_R2",
                    "local_node_connectivity",
                    "mean_degree_mixing_matrix",
                    "nb_edges",
                    "nb_nodes",
                    "node_connectivity" ,
                    "resource_allocation_index", "treewidth_min_degree",
                    "add_closeness_centrality"]

metric_sort_val = {}
metric_dataloader_val = {}

#@profile
def sort_metric_dataset_val(df_val_metric,args):
    
    global metric_sort_val
    global metric_to_consider

    for m in metric_to_consider:
        if "A" in args.metric_order:
            metric_sort_val[m + "_A"] = df_val_metric.sort_values(by=m,ascending=True).index.tolist()
        
        if "D" in args.metric_order:
            metric_sort_val[m + "_D"] = df_val_metric.sort_values(by=m,ascending=False).index.tolist()
        
        if "Q" in args.metric_order:
            complexity_scores = df_val_metric[m].to_numpy()
            data = QuantileTransformer(n_quantiles=len(complexity_scores), output_distribution='normal', random_state=0).fit_transform(complexity_scores.reshape(-1, 1))
            data = np.absolute(data - data.mean())
            #pdf = norm.pdf(data)  # probability of transformed samples
            tmp = []
            for (k,i, j) in (sorted(zip(range(len(complexity_scores)), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=True)):
                tmp.append(k)
            metric_sort_val[m + "_Qd"] = tmp
            
            tmp = []
            for (k,i, j) in (sorted(zip(range(len(complexity_scores)), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=False)):
                tmp.append(k)
            metric_sort_val[m + "_Qa"] = tmp
        

    
    if args.add_random:
        if "A" in args.metric_order:
            metric_sort_val["random_A"] = df_val_metric.sort_values(by="random",ascending=True).index.tolist()
        if "D" in args.metric_order:
            
            metric_sort_val["random_D"] = df_val_metric.sort_values(by="random",ascending=False).index.tolist()
        if "Q" in args.metric_order:
            complexity_scores = df_val_metric["random"]
            data = QuantileTransformer(n_quantiles=len(complexity_scores), output_distribution='normal', random_state=0).fit_transform(complexity_scores.reshape(-1, 1))
            data = np.absolute(data - data.mean())
            #pdf = norm.pdf(data)  # probability of transformed samples
            tmp = []
            for (k,i, j) in (sorted(zip(range(n_samples), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=True)):
                tmp.append(k)
            metric_sort_val["random_Qd"] = tmp
            
            tmp = []
            for (k,i, j) in (sorted(zip(range(n_samples), complexity_scores, data.flatten()), key=lambda x: x[2], reverse=False)):
                tmp.append(k)
            metric_sort_val["random_Qa"] = tmp
            
    if args.sum_of_all_metric:
            metric_sort_val["sum_of_all_metric_A"] = df_val_metric.sort_values(by="sum_of_all_metric",ascending=True).index.tolist()



#@profile
def trim_dataset_as_per_competency_val(val_set, metric, c, order,forward_sample_size):
    if "A" == order:
            key = metric + "_A"
    if "D" ==order:
            
            key = metric + "_D"
    if "Qa"== order:
            key = metric + "_Qa"
    if "Qd"== order:
            key = metric + "_Qd"
    D = metric_sort_val[key]
    nb_examples = int(c * len(val_set))
    c_idx = D[:nb_examples]
    c_idx = random.sample(c_idx, forward_sample_size) if len(c_idx) > forward_sample_size else c_idx
    
    return [val_set[i] for i in c_idx]



#@profile
def revise_metric_dataloader_val(bs, val_set, c, args, forward_sample_size):
    global metric_dataloader_val
    global metric_to_consider

    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            
            metric_dataloader_val[m + "_A"] = DataLoader(trim_dataset_as_per_competency_val(val_set, m, c, "A",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
        if "D" in args.metric_order:
            
            metric_dataloader_val[m + "_D"] = DataLoader(trim_dataset_as_per_competency_val(val_set, m, c, "D",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
        if "Qa" in args.metric_order:
            
            metric_dataloader_val[m + "_Qa"] = DataLoader(trim_dataset_as_per_competency_val(val_set, m, c, "Qa",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        if "Qd" in args.metric_order:
            
            metric_dataloader_val[m + "_Qd"] = DataLoader(trim_dataset_as_per_competency_val(val_set, m, c, "Qd",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
    
    if args.add_random:
        metric_dataloader_val["random" + "_Qa"] = DataLoader(trim_dataset_as_per_competency_val(val_set, "random", c, "Qa",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        metric_dataloader_val["random" + "_Qd"] = DataLoader(trim_dataset_as_per_competency_val(val_set, "random", c, "Qd",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        metric_dataloader_val["random" + "_A"] = DataLoader(trim_dataset_as_per_competency_val(val_set, "random", c, "A",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        metric_dataloader_val["random" + "_D"] = DataLoader(trim_dataset_as_per_competency_val(val_set, "random", c, "D",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

    if args.sum_of_all_metric:
        metric_dataloader_val["sum_of_all_metric" + "_A"] = DataLoader(trim_dataset_as_per_competency_val(val_set, "sum_of_all_metric", c, "A",forward_sample_size), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

#@profile
def revise_metric_dict_val(args):

        
    asc_dict = {}
    desc_dict = {}
    q_dict = {}
    metric_dict = {}
    if args.use_k_means:
        if args.dataset == "pgr":
                
            pgr_k_means_metric = ["add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "degree", "degree_assortativity_coefficient", "density",
                                  "len_local_bridges", "mean_degree_mixing_matrix", "node_connectivity", "treewidth_min_degree", "add_closeness_centrality"]
            for m in pgr_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader_val[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader_val[m + "_D"]
                if "Qa" in args.metric_order:
                    q_dict[m +"_Qa"] = metric_dataloader_val[m + "_Qa"]
                if "Qd" in args.metric_order:
                    q_dict[m +"_Qd"] = metric_dataloader_val[m + "_Qd"]
                    
        elif args.dataset == "omim":
            omim_k_means_metric = ["add_avg_neighbor_deg", "add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "large_clique_size", "degree_assortativity_coefficient",
                                  "density", "add_katz_centrality_numpy", "group_degree_centrality", "treewidth_min_degree", "add_closeness_centrality"]
            for m in omim_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader_val[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader_val[m + "_D"]
                if "Qa" in args.metric_order:
                    q_dict[m +"_Qa"] = metric_dataloader_val[m + "_Qa"]
                if "Qd" in args.metric_order:
                    q_dict[m +"_Qd"] = metric_dataloader_val[m + "_Qd"]
                    
    else:
        
        for m in metric_to_consider:
            if "A" in args.metric_order:
                asc_dict[m +"_A"] = metric_dataloader_val[m + "_A"]
            if "D" in args.metric_order:
                desc_dict[m +"_D"] = metric_dataloader_val[m + "_D"]
            if "Qa" in args.metric_order:
                q_dict[m +"_Qa"] = metric_dataloader_val[m + "_Qa"]
            if "Qd" in args.metric_order:
                q_dict[m +"_Qd"] = metric_dataloader_val[m + "_Qd"]
                
    if args.add_random:
        
            print("random added")
            if "A" in args.metric_order:
                asc_dict["random" +"_A"] = metric_dataloader_val["random" + "_A"]
            if "D" in args.metric_order:
                 desc_dict["random" +"_D"] = metric_dataloader_val["random" + "_D"]
            if "Qa" in args.metric_order:
                 q_dict["random" +"_Qa"] = metric_dataloader_val["random" + "_Qa"]
            if "Qd" in args.metric_order:
                 q_dict["random" +"_Qd"] = metric_dataloader_val["random" + "_Qd"]


    if args.sum_of_all_metric:
            sum_dict = {}
            print("sum_of_all_metric")
            if "A" in args.metric_order:
                sum_dict["sum_of_all_metric" +"_A"] = metric_dataloader_val["sum_of_all_metric" + "_A"]
    
    if args.sum_of_all_metric:
        metric_dict.update(sum_dict)
    else:
        metric_dict.update(asc_dict)
        metric_dict.update(desc_dict)
        metric_dict.update(q_dict)
                
    
    
    return metric_dict
