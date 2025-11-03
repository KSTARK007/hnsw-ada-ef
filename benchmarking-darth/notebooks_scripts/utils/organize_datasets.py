import numpy as np
import faiss
import sys
import os
import h5py
import hnswlib

from concurrent.futures import ThreadPoolExecutor
import time

DARTH_ROOT = os.environ["DARTH_ROOT"]       
DIRECTORY_TO_LOAD = os.path.join(DARTH_ROOT, "datasets")
DIRECTORY_TO_SAVE = os.path.join(DARTH_ROOT, "datasets/processed")

os.makedirs(f"{DIRECTORY_TO_SAVE}", exist_ok=True)

# File writers
def write_ivecs(file_name, indices):
    with open(file_name, 'wb') as f:
        for vec in indices:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.int32).tobytes())
            
def write_fvecs(file_name, vectors):
    with open(file_name, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(np.array(vec, dtype=np.float32).tobytes())

# File readers
def read_bvecs(file_path, limit=None):
    vectors = None
    dim = None
    with open(file_path, 'rb') as f:
        first_entry = np.fromfile(f, dtype=np.int32, count=1)
        if len(first_entry) == 0:
            raise ValueError("The file is empty or not in the expected bvecs format.")
        dim = first_entry[0]

        f.seek(0)

        vector_size = np.int64(4 + dim)  # 4 bytes for int32 + dim bytes for vector data
        file_size = np.int64(f.seek(0, 2))  # Ensure file_size is an int
        file_size = f.tell()
        f.seek(0)

        num_vectors = np.int64(np.int64(file_size) // np.int64(vector_size))

        print(f">>Dimension: {dim}, Total Num vectors: {num_vectors}")
        
        if limit is not None:
            num_vectors = min(limit, num_vectors)
            
        print(f">>Limiting to {num_vectors} vectors")

        print(type(num_vectors))
        print(type(vector_size))
        print(type(num_vectors * vector_size))
        
        data = np.fromfile(f, dtype=np.uint8, count=num_vectors * vector_size)
        
        print(f">>Data shape: {data.shape}")
        
        data = data.reshape(-1, vector_size)
        
        vectors = data[:, 4:]
        assert vectors.shape[1] == dim  # Ensure shape consistency

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

        vector_size = np.int64((dim + 1) * 4)  # 4 bytes per float

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
        
        
        n = np.int64(n)
        dim = np.int64(dim)
        vectors = np.fromfile(f, count=np.int64(n * dim), dtype=np.float32)
        vectors = vectors.reshape(n, dim)
    
    return vectors, int(dim)
    
dataset_info = {
    "deepimage" : {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/deepimage/deepimage_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/deepimage/deepimage_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/deepimage/deepimage_query.fvecs",
        
        "base_vectors_to_load": 9990000,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 1000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/deepimage/base.9.99M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/deepimage/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/learn.groundtruth.10K.k100.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/deepimage/validation.groundtruth.1K.k100.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/query.groundtruth.10K.k100.ivecs",

        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/learn.groundtruth.10K.k100.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/deepimage/validation.groundtruth.1K.k100.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/deepimage/query.groundtruth.10K.k100.fvecs",

        "k": 100,
        "d": 96,
        "read_function": read_fvecs
    },
    
    "glove100": {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/glove100/glove100_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/glove100/glove100_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/glove100/glove100_query.fvecs",
        
        "base_vectors_to_load": 1183514,        
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 1000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/glove100/base.1.18M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/glove100/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/learn.groundtruth.10K.k100.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/glove100/validation.groundtruth.1K.k100.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/query.groundtruth.10K.k100.ivecs",
       
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/learn.groundtruth.10K.k100.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/glove100/validation.groundtruth.1K.k100.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/glove100/query.groundtruth.10K.k100.fvecs",
        
        "k": 100,
        "d": 100,
        "read_function": read_fvecs
    },
    
    "msmarco_v1" : {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/msmarco_v1/msmarco_v1_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/msmarco_v1/msmarco_v1_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/msmarco_v1/msmarco_v1_query.fvecs",

        "base_vectors_to_load": 8841823,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 6980,
        
        "validation_vectors_to_generate": 1000,        
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/msmarco_v1/base.8.84M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v1/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/query.6.98K.fvecs",

        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v1/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/query.groundtruth.6.98K.k1000.ivecs",

        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v1/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v1/query.groundtruth.6.98K.k1000.fvecs",

        "k": 1000,
        "d": 1536,
        "read_function": read_fvecs
    },
    
    "msmarco_v2.1" : {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/msmarco_v2.1/msmarco_v2.1_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/msmarco_v2.1/msmarco_v2.1_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/msmarco_v2.1/msmarco_v2.1_query.fvecs",

        "base_vectors_to_load": 18380565,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 1677,
        
        "validation_vectors_to_generate": 1000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/base.18.38M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/query.1.67K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/query.groundtruth.1.67K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/msmarco_v2.1/query.groundtruth.1.67K.k1000.fvecs",
        
        "k": 1000,
        "d": 1024,
        "read_function": read_fvecs
    },
    
    "laion_i2i" : {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/laion_i2i/laion_i2i_base.fvecs",
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/laion_i2i/laion_i2i_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/laion_i2i/laion_i2i_query.fvecs",

        "base_vectors_to_load": 30646115,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        
        "validation_vectors_to_generate": 1000,

        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/laion_i2i/base.30.65M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/laion_i2i/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/laion_i2i/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/laion_i2i/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/laion_i2i/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 512,
        "read_function": read_fvecs
    },
    
    "laion_t2i" : {
        "provided_base_filepath":  f"{DIRECTORY_TO_LOAD}/laion_t2i/laion_t2i_base.fvecs", 
        "provided_learn_filepath": f"{DIRECTORY_TO_LOAD}/laion_t2i/laion_t2i_learn.fvecs",
        "provided_query_filepath": f"{DIRECTORY_TO_LOAD}/laion_t2i/laion_t2i_query.fvecs",

        "base_vectors_to_load": 30646115,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,

        "validation_vectors_to_generate": 1000,

        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/laion_t2i/base.30.65M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/laion_t2i/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/query.10K.fvecs",

        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/laion_t2i/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/query.groundtruth.10K.k1000.ivecs",

        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/laion_t2i/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/laion_t2i/query.groundtruth.10K.k1000.fvecs",

        "k": 1000,
        "d": 512,
        "read_function": read_fvecs
    },
    
    "uniform_cluster" : {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/uniform_cluster/uniform_cluster_base.fvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/uniform_cluster/uniform_cluster_learn.fvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/uniform_cluster/uniform_cluster_query.fvecs",

        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,

        "validation_vectors_to_generate": 1000,

        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/uniform_cluster/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/uniform_cluster/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/uniform_cluster/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/uniform_cluster/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/uniform_cluster/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 100,
        "read_function": read_fvecs        
    },
    
    "zipfian_cluster": {
        "provided_base_filepath":   f"{DIRECTORY_TO_LOAD}/zipfian_cluster/zipfian_cluster_base.fvecs",
        "provided_learn_filepath":  f"{DIRECTORY_TO_LOAD}/zipfian_cluster/zipfian_cluster_learn.fvecs",
        "provided_query_filepath":  f"{DIRECTORY_TO_LOAD}/zipfian_cluster/zipfian_cluster_query.fvecs",
        
        "base_vectors_to_load": 10000000,
        "learn_vectors_to_load": 10000,
        "query_vectors_to_load": 10000,
        "validation_vectors_to_generate": 1000,
        
        "processed_base_filepath":          f"{DIRECTORY_TO_SAVE}/zipfian_cluster/base.10M.fvecs",
        "processed_learn_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/learn.10K.fvecs",
        "processed_validation_filepath":    f"{DIRECTORY_TO_SAVE}/zipfian_cluster/validation.1K.fvecs",
        "processed_query_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/query.10K.fvecs",
        
        "processed_learn_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/learn.groundtruth.10K.k1000.ivecs",
        "processed_validation_groundtruth_filepath":    f"{DIRECTORY_TO_SAVE}/zipfian_cluster/validation.groundtruth.1K.k1000.ivecs",
        "processed_query_groundtruth_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/query.groundtruth.10K.k1000.ivecs",
        
        "processed_learn_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/learn.groundtruth.10K.k1000.fvecs",
        "processed_validation_gtdistances_filepath":    f"{DIRECTORY_TO_SAVE}/zipfian_cluster/validation.groundtruth.1K.k1000.fvecs",
        "processed_query_gtdistances_filepath":         f"{DIRECTORY_TO_SAVE}/zipfian_cluster/query.groundtruth.10K.k1000.fvecs",
        
        "k": 1000,
        "d": 100,
        "read_function": read_fvecs
    }        
}

def organize_dataset(ds_name):
    os.makedirs(f"{DIRECTORY_TO_SAVE}/{ds_name}/", exist_ok=True)
    
    print(f">> Organizing dataset: {ds_name}")
    cpu_count = os.cpu_count() or 1
    num_threads = max(1, cpu_count // 4)
    faiss.omp_set_num_threads(num_threads)
    print(f">> Using {num_threads} threads for Faiss")
    
    k = dataset_info[ds_name]["k"]
    d = dataset_info[ds_name]["d"]
    read_function = dataset_info[ds_name]["read_function"]
    
    # Save the processed vectors
    base_vectors_num = dataset_info[ds_name]["base_vectors_to_load"]
    base_vectors, dim = read_function(dataset_info[ds_name]["provided_base_filepath"], limit=base_vectors_num)
    assert dim == d and len(base_vectors) == base_vectors_num, f"Base vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, base_vectors.shape: {base_vectors.shape}")
    
    query_vectors_num = dataset_info[ds_name]["query_vectors_to_load"]
    query_vectors, dim = read_function(dataset_info[ds_name]["provided_query_filepath"], limit=query_vectors_num)
    assert dim == d and len(query_vectors) == query_vectors_num, f"Query vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, query_vectors.shape: {query_vectors.shape}")
    
    learn_vectors_num = dataset_info[ds_name]["learn_vectors_to_load"]
    validation_vectors_num = dataset_info[ds_name]["validation_vectors_to_generate"]
    learn_and_validation_vectors_num = learn_vectors_num + validation_vectors_num
    learn_and_validation_vectors, dim = read_function(dataset_info[ds_name]["provided_learn_filepath"], limit=learn_and_validation_vectors_num)
    assert dim == d and len(learn_and_validation_vectors) == learn_and_validation_vectors_num, f"Learn and validation vectors shape mismatch for {ds_name}"
    print(f">> ds_name: {ds_name}, learn_and_validation_vectors.shape: {learn_and_validation_vectors.shape}")
    
    learn_vectors = learn_and_validation_vectors[:learn_vectors_num]
    validation_vectors = learn_and_validation_vectors[learn_vectors_num:]
    
    write_fvecs(dataset_info[ds_name]["processed_base_filepath"], base_vectors)
    write_fvecs(dataset_info[ds_name]["processed_query_filepath"], query_vectors)
    write_fvecs(dataset_info[ds_name]["processed_learn_filepath"], learn_vectors)
    write_fvecs(dataset_info[ds_name]["processed_validation_filepath"], validation_vectors)
    
    # temp: start
    base_vectors = read_function(dataset_info[ds_name]["processed_base_filepath"])[0]
    learn_vectors = read_function(dataset_info[ds_name]["processed_learn_filepath"])[0]
    validation_vectors = read_function(dataset_info[ds_name]["processed_validation_filepath"])[0]
    query_vectors = read_function(dataset_info[ds_name]["processed_query_filepath"])[0]
    print(f">> Loaded processed vectors for {ds_name}")
    print(f">> base_vectors.shape: {base_vectors.shape}")
    print(f">> learn_vectors.shape: {learn_vectors.shape}")
    print(f">> validation_vectors.shape: {validation_vectors.shape}")
    print(f">> query_vectors.shape: {query_vectors.shape}")
    # temp: end
    
    base_vectors = np.ascontiguousarray(base_vectors, dtype='float32')
    learn_vectors = np.ascontiguousarray(learn_vectors, dtype='float32')
    validation_vectors = np.ascontiguousarray(validation_vectors, dtype='float32')
    query_vectors = np.ascontiguousarray(query_vectors, dtype='float32')
    
    print(">> Normalizing vectors for cosine similarity...")
    faiss.normalize_L2(base_vectors)
    faiss.normalize_L2(learn_vectors)
    faiss.normalize_L2(validation_vectors)
    faiss.normalize_L2(query_vectors)
    
    # Calculate the groundtruth using FAISS
    # the exact search based faiss is extremely low and does not scale to multithreaded, see: https://github.com/facebookresearch/faiss/issues/4121
    # as DARTH requires computing ground truth of 10k queries, this would not be feasible with faiss therefore, we use hnswlib to compute ground truth
    # index = faiss.IndexFlatIP(d)     
    # start_time = time.time()
    # index.add(base_vectors) #type: ignore
    # print(f">> Index add time: {time.time() - start_time:.3f} seconds")
    
    bf = hnswlib.BFIndex(space='ip', dim=d)
    bf.init_index(max_elements=base_vectors.shape[0])
    bf.set_num_threads(num_threads)
    start_time = time.time()
    bf.add_items(base_vectors)
    print(f">> Index add time: {time.time() - start_time:.3f} seconds") 

    start_time = time.time()
    # distances, indices = index.search(learn_vectors, k)  #type: ignore
    indices, distances = bf.knn_query(learn_vectors, k=k)
    distances = 1.0 - distances  # convert cosine distance to cosine similarity, as faiss uses similarity    
    print(f">> Index search time for {learn_vectors_num} learn vectors: {time.time() - start_time:.3f} seconds")
    assert indices.shape == (learn_vectors_num, k), f"Learn groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (learn_vectors_num, k), f"Learn groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_learn_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_learn_gtdistances_filepath"], distances)
    
    start_time = time.time()
    # distances, indices = index.search(validation_vectors, k)  #type: ignore
    indices, distances = bf.knn_query(validation_vectors, k=k)
    distances = 1.0 - distances  # convert cosine distance to cosine similarity, as faiss uses similarity
    print(f">> Index search time for {validation_vectors_num} validation vectors: {time.time() - start_time:.3f} seconds")
    assert indices.shape == (validation_vectors_num, k), f"Validation groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (validation_vectors_num, k), f"Validation groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_validation_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_validation_gtdistances_filepath"], distances)

    start_time = time.time()
    # distances, indices = index.search(query_vectors, k)  #type: ignore
    indices, distances = bf.knn_query(query_vectors, k=k)
    distances = 1.0 - distances  # convert cosine distance to cosine similarity, as faiss uses similarity    
    print(f">> Index search time for {query_vectors_num} query vectors: {time.time() - start_time:.3f} seconds")
    assert indices.shape == (query_vectors_num, k), f"Query groundtruth shape mismatch for {ds_name}"
    assert distances.shape == (query_vectors_num, k), f"Query groundtruth distances shape mismatch for {ds_name}"
    write_ivecs(dataset_info[ds_name]["processed_query_groundtruth_filepath"], indices)
    write_fvecs(dataset_info[ds_name]["processed_query_gtdistances_filepath"], distances)

    print(">> Done organizing dataset: ", ds_name)


def process_hdf5(file_path, dataset, num_learn, num_validation, file_path2=None):
    """Helper for converting an HDF5 dataset into fvecs files under DIRECTORY_TO_LOAD/dataset."""
    print(f"Processing HDF5 dataset: {dataset} from {file_path}")
    
    path = os.path.expanduser(file_path)
    with h5py.File(path, "r") as f:
        base = np.array(f["train"])      
        if dataset != "laion_t2i":
            query = np.array(f["test"])
    
    if dataset == "laion_t2i" and file_path2 is not None: # special case for laion_t2i where test set is in a different file
        path = os.path.expanduser(file_path2) 
        with h5py.File(path, "r") as f:
            query = np.array(f["test"])

    np.random.seed(987)
    total = int(num_learn) + int(num_validation)
    if total > base.shape[0]:
        raise ValueError("Requested learn+validation vectors exceed base size")
    learn_indices = np.random.choice(base.shape[0], size=total, replace=False)
    learn = base[learn_indices]

    if dataset != "laion_t2i": # laion_t2i does not use the original text base set; instead, it uses the image base set
        print(f"base shape: {base.shape}")
    print(f"query shape: {query.shape}") #type: ignore
    print(f"learn shape: {learn.shape}")

    out_dir = os.path.join(DIRECTORY_TO_LOAD, dataset)
    os.makedirs(out_dir, exist_ok=True)
    
    write_fvecs(os.path.join(out_dir, f"{dataset}_learn.fvecs"), learn)
    
    if dataset != "laion_t2i": # laion_t2i does not use the text original base set; instead, it uses the image base set
        write_fvecs(os.path.join(out_dir, f"{dataset}_base.fvecs"), base)
    else:
        print("copying laion_i2i base as laion_t2i base")
        os.system(f"cp {DIRECTORY_TO_LOAD}/laion_i2i/laion_i2i_base.fvecs {out_dir}/laion_t2i_base.fvecs")
        
    write_fvecs(os.path.join(out_dir, f"{dataset}_query.fvecs"), query) #type: ignore
    
    

ADA_EF_ROOT = os.getenv("ADA_EF_ROOT", "")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/deep-image-96-angular.hdf5", "deepimage", 10000, 1000)
organize_dataset("deepimage")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/glove-100-angular.hdf5", "glove100", 10000, 1000)
organize_dataset("glove100")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/msmarco.hdf5", "msmarco_v1", 10000, 1000)
organize_dataset("msmarco_v1")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/cohere.hdf5", "msmarco_v2.1", 10000, 1000)
organize_dataset("msmarco_v2.1")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/laion_image.hdf5", "laion_i2i", 10000, 1000)
organize_dataset("laion_i2i")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/laion_text_ng.hdf5", "laion_t2i", 10000, 1000, f"{ADA_EF_ROOT}/experiments/data/laion_text_query_groundtruth.hdf5")
organize_dataset("laion_t2i")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/cluster_mg_uniform_100d.hdf5", "uniform_cluster", 10000, 1000)
organize_dataset("uniform_cluster")

process_hdf5(f"{ADA_EF_ROOT}/experiments/data/cluster_mg_zipf_100d.hdf5", "zipfian_cluster", 10000, 1000)
organize_dataset("zipfian_cluster")
