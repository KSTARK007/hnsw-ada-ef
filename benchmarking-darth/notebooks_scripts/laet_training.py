import os
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import dask.dataframe as dd

SEED = 42

def _series_memory_bytes(series: pd.Series) -> int:
    if series.empty:
        return 0
    return len(series) * 4 # assuming float32 or int32

def _dataframe_memory_bytes(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    row_count = len(df)
    column_bytes = sum(4 for _ in df.dtypes) # assuming float32 or int32
    return row_count * column_bytes

def compute_memory_footprint(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[int, int]:
    return _dataframe_memory_bytes(X_train), _series_memory_bytes(y_train)

def get_dataset_name(M, efC, efS, query_num, ds_name, k, logint):    
    return f"{os.environ['DARTH_ROOT']}/et_training_data/early-stop-training/{ds_name}/k{k}/M{M}_efC{efC}_efS{efS}_qs{query_num}_li{logint}.txt"

def get_optimal_dataset_name(M, ef, s, ds_name, k, logint):
    return f"../results/early-stop-training/RECALL_{ds_name}_M{M}_ef{ef}_k{k}_qs{s}_li{logint}.txt"

def read_bvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected bvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = 4 + dim

        file_size = f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        num_vectors = file_size // vector_size
        
        #print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")
        
        if (limit is not None) and limit < num_vectors:
            num_vectors = limit
            #print(f">>Limiting to {num_vectors} vectors") 

        data = np.fromfile(f, dtype=np.uint8, count=num_vectors * vector_size)
        data = data.reshape(-1, vector_size)
        
        vectors = data[:, 4:]
        
        #print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)
    
    return vectors, int(dim)

def read_fvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected fvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = (dim + 1) * 4  # 4 bytes per float

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)

        num_vectors = file_size // vector_size

        #print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")

        if limit is not None and limit < num_vectors:
            num_vectors = limit
            #print(f">>Limiting to {num_vectors} vectors")

        data = np.fromfile(f, dtype=np.float32, count=num_vectors * (dim + 1))
        data = data.reshape(-1, dim + 1)

        vectors = data[:, 1:]

        #print(f">>Vectors shape: {vectors.shape}")
        assert vectors.shape == (num_vectors, dim)

    return vectors, int(dim)

def read_fbin_vecs(file_path, limit=None):
    vectors = None
    dim = None
    
    with open(file_path, 'rb') as f:
        n = np.fromfile(f, count=1, dtype=np.uint32)[0]
        dim = np.fromfile(f, count=1, dtype=np.uint32)[0]
        
        if limit is not None and n > limit:
            n = limit
            #print(f">> Limiting dataset {file_path} to {n} vectors")
        
        vectors = np.fromfile(f, count=n * dim, dtype=np.float32)
        vectors = vectors.reshape(n, dim)
    
    return vectors, int(dim)

def compute_P99(y_true, y_pred):
    y_diff = np.abs(y_true - y_pred)
    return np.percentile(y_diff, 99)

def get_query_dims_df(data_df, d, ds_name, s):
    dimensions = np.arange(1, d + 1)
    per_query_dimensions_df = pd.DataFrame(index=data_df["qid"].unique(), columns=[f"d{d}" for d in dimensions])

    query_dims = None
    dims_read = None
    
    directory_to_load = f"{os.environ['DARTH_ROOT']}/datasets/processed"
    if ds_name == "deepimage":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/deepimage/learn.10K.fvecs", limit=s)
    elif ds_name == "glove100":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/glove100/learn.10K.fvecs", limit=s)
    elif ds_name == "msmarco_v1":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/msmarco_v1/learn.10K.fvecs", limit=s)
    elif ds_name == "msmarco_v2.1":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/msmarco_v2.1/learn.10K.fvecs", limit=s)
    elif ds_name == "laion_i2i":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/laion_i2i/learn.10K.fvecs", limit=s)
    elif ds_name == "laion_t2i":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/laion_t2i/learn.10K.fvecs", limit=s)
    elif ds_name == "uniform_cluster":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/uniform_cluster/learn.10K.fvecs", limit=s)
    elif ds_name == "zipfian_cluster":
        query_dims, dims_read = read_fvecs(f"{directory_to_load}/zipfian_cluster/learn.10K.fvecs", limit=s)

    assert query_dims.shape[0] == s, f"Expected {s} queries, got {query_dims.shape[0]}" # type: ignore
    assert query_dims.shape[1] == d, f"Expected {d} dimensions, got {query_dims.shape[1]}" # type: ignore
    assert dims_read == d, f"Expected {d} dimensions, got {dims_read}"
    
    # Make sure that this is correct
    for i, qid in enumerate(data_df["qid"].unique()):
        per_query_dimensions_df.loc[qid] = query_dims[i]  # type: ignore

    #assert len(per_query_dimensions_df) == s, f"Expected {s} queries, got {len(per_query_dimensions_df)}"
    
    return per_query_dimensions_df, dimensions
        
def main():
    s, li = 10000, 2

    dataset_info = {
        "deepimage": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 96,
            "F": 450,
            "k": 100,
        },
        "glove100": {
            "M": 16,
            "efC": 500,
            "efS": 500,
            "d": 100,
            "F": 1362,
            "k": 100,
        },
        "msmarco_v1": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 1536,
            "F": 1504,
            "k": 1000,
        },
        "msmarco_v2.1": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 1024,
            "F": 1704,
            "k": 1000,
        },
        "uniform_cluster": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 100,
            "F": 1094,
            "k": 1000,
        },
        "zipfian_cluster": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 100,
            "F": 4844,
            "k": 1000,
        },
        "laion_i2i": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 512,
            "F": 1920,
            "k": 1000,
        },
        "laion_t2i": {
            "M": 16,
            "efC": 500,
            "efS": 2000,
            "d": 512,
            "F": 4208,
            "k": 1000,
        }
    }

    for k in [0]: # this is a dummy loop to allow easy indentation
        for ds_name in dataset_info.keys():        
            d = dataset_info[ds_name]["d"]
            F = dataset_info[ds_name]["F"]
            M = dataset_info[ds_name]["M"]
            efC = dataset_info[ds_name]["efC"]
            efS = dataset_info[ds_name]["efS"]
            k = dataset_info[ds_name]["k"] # actual k value
            print (f"Training for {ds_name} with k={k}")
            
            # Load all the data
            columns_to_load = ["qid", "dists", "first_nn_dist", "nn_dist", "nn10_dist", "nn_to_first", "nn10_to_first", "r"]
            dask_df = dd.read_csv(get_dataset_name(M, efC, efS, 10000, ds_name, k, li), usecols=columns_to_load) #type: ignore
            data_df = dask_df.compute()
            data_df = data_df[data_df["qid"] < s]
            
            # Some queries (usually 1-2) might finish early and not have all the data, we skip them
            #missing_queries = set(range(s)) - set(data_df["qid"].unique())
            #print(f"Missing queries: {missing_queries} (len: {len(missing_queries)})")        
            
            total_queries = len(data_df["qid"].unique())  
            print(f"Total queries: {total_queries}")      
    
            # Load the dimensions of the queries
            per_query_dimensions_df, dimensions = get_query_dims_df(data_df, d, ds_name, s)
            assert len(per_query_dimensions_df) == total_queries, f"1. Expected {total_queries} queries, got {len(per_query_dimensions_df)}"
            
            # Set the features to be used as input
            query_dims_features = [f"d{d}" for d in dimensions]
            input_features = ["first_nn_dist", "nn_dist", "nn10_dist", "nn_to_first", "nn10_to_first"] + query_dims_features

            # Compute the target variables for each recall target
            data_recall_log_df = data_df#pd.read_csv(get_optimal_dataset_name(M, ef, s, ds_name, k, li), usecols=["qid", "r", "dists"], low_memory=False)
            per_query_target_df = pd.DataFrame(index=data_recall_log_df["qid"].unique(), columns=[f"dists_for_max_recall"])
            for qid, q_df in data_recall_log_df.groupby("qid"):
                per_query_target_df.at[qid, f"dists_for_max_recall"] = q_df[q_df["r"] == q_df["r"].max()]["dists"].min() 
            
            assert len(per_query_target_df) == total_queries, f"2. Expected {total_queries} queries, got {len(per_query_target_df)}"
            
            print(per_query_target_df)
            
            # Pick only the input for each query after F distance calcs
            per_query_input_feats_df = data_df[data_df["dists"] == F]
                    
            # Merge the inputs with the target
            per_query_input_feats_df = per_query_input_feats_df.merge(per_query_target_df, left_on="qid", right_index=True)
            #assert len(per_query_input_feats_df) == total_queries, f"4. Expected {total_queries} queries, got {len(per_query_input_feats_df)}"
                    
            # Merge the inputs with the query dimensions
            per_query_input_feats_df = per_query_input_feats_df.merge(per_query_dimensions_df, left_on="qid", right_index=True)
            #assert len(per_query_input_feats_df) == total_queries, f"5. Expected {total_queries} queries, got {len(per_query_input_feats_df)}"
                    
            # drop all rows where the target is NaN
            per_query_input_feats_df = per_query_input_feats_df.dropna(subset=[f"dists_for_max_recall"])
                    
            X = per_query_input_feats_df[input_features].values
            y = per_query_input_feats_df["dists_for_max_recall"].values
                
            X_train, y_train = X, y
            # print shape of x and y
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            x_memory_bytes, y_memory_bytes = compute_memory_footprint(per_query_input_feats_df[input_features], per_query_input_feats_df["dists_for_max_recall"])
                    
            model = lgb.LGBMRegressor(objective='regression', random_state=SEED)
            model_train_time_start = time.time()
            model.fit(X_train, y_train)
            model_train_time = time.time() - model_train_time_start
        
            # create directory if not exists
            os.makedirs(f"{os.environ['DARTH_ROOT']}/predictor_models/laet/", exist_ok=True)
            model_file = f"{os.environ['DARTH_ROOT']}/predictor_models/laet/{ds_name}_M{M}_efC{efC}_efS{efS}_s{s}_k{k}.txt"
            model.booster_.save_model(model_file)
            model_memory_bytes = os.path.getsize(model_file)
            print(f"Model training time: {model_train_time}")
            print(f"X memory bytes: {x_memory_bytes}, y memory bytes: {y_memory_bytes}, model memory bytes: {model_memory_bytes}")

if __name__ == "__main__":
    main()
