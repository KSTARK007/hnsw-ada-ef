import pandas as pd
import numpy as np
import json
import os

import dask.dataframe as dd

def get_dataset_name(M, efC, efS, s, ds_name, k, logint):
    return f"{os.environ['DARTH_ROOT']}/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{s}_li{logint}.txt"

def extract_data_stats(data):    
    results = {}
    
    target_recalls = [0.5, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99, 1.00]
    grouped_by_query = data.groupby("qid")
    
    results["correlations_to_target"] = {}
    correlations_to_target = {}
    for col in data.columns:
        if col != "r":
            correlations_to_target[col] = data["r"].corr(data[col])
    
    correlations_to_target = {k: v for k, v in sorted(correlations_to_target.items(), key=lambda item: abs(item[1]), reverse=True)}
    for col, corr in correlations_to_target.items():
        #print(f"        {col}: {corr:.4f}")
        results["correlations_to_target"][col] = corr
    
    results["distance_calcs_to_reach_target_recall"] = {}
    for target_recall in target_recalls:
        distance_calcs_to_reach_target_recall = []
        for idx, group in grouped_by_query:
            group = group[group["r"] >= target_recall]
            if group.shape[0] > 0:
                row_of_first_reached_recall = group.iloc[0]
                distance_calcs_to_reach_target_recall.append(row_of_first_reached_recall["dists"])
        
        mean_dists = np.mean(distance_calcs_to_reach_target_recall)
        max_dists = np.max(distance_calcs_to_reach_target_recall)
        min_dists = np.min(distance_calcs_to_reach_target_recall)
        results["distance_calcs_to_reach_target_recall"][target_recall] = {
            "avg": mean_dists,
            "max": max_dists,
            "min": min_dists
        }
    
    return results

dataset_params = {        
        "deepimage": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "k": 100,
        },        
        "glove100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "k": 100,   
        },        
        "msmarco_v1": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,   
        },        
        "msmarco_v2.1": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,   
        },
        "uniform_cluster": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,   
        },        
        "zipfian_cluster": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,   
        },
        "laion_i2i": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,   
        },
        "laion_t2i": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "k": 1000,
        }        
    }

index_metric_feats = ["step", "dists", "inserts"]
neighbor_distances_feats = ["first_nn_dist", "nn_dist", "furthest_dist"]
neighbor_stats_feats = ["avg_dist", "variance", "percentile_25", "percentile_50", "percentile_75"]
all_feats = index_metric_feats + neighbor_distances_feats + neighbor_stats_feats
columns_to_load = ["qid", "elaps_ms"] + all_feats + ["r", "feats_collect_time_ms"]


all_k_values = [0] # dummy k value, as we use different k values for different datasets

for ds_name in dataset_params.keys():
    dataset_stats = {}
    
    M = dataset_params[ds_name]["M"]
    efC = dataset_params[ds_name]["efC"]
    efS = dataset_params[ds_name]["efS"]
    for k in all_k_values:
        k = dataset_params[ds_name]["k"] # actual k value
        print(f"{ds_name} | k={k}")
        li = 2
        dask_df = dd.read_csv(get_dataset_name(M, efC, efS, 10000, ds_name, k, li), usecols=columns_to_load) #type: ignore
        data_all = dask_df.compute()
        
        dataset_stats[k] = extract_data_stats(data_all)
    
    with open(f"./data_stats_{ds_name}.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)          
    