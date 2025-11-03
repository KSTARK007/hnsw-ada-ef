#include "VectorDataLoader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

const char query_type_str[4][20] = {
    "Training",
    "Validation",
    "Testing",
    "Noisy Testing"
};

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out,
        size_t limit) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        exit(1);
    }

    int d;
    fread(&d, 1, sizeof(int), f);

    assert((d > 0 && d < 1000000) && "unreasonable dimension");

    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    if (limit > 0 && n > limit) {
        n = limit;
        printf("Limiting dataset %s to %ld vectors\n", fname, n);
    }

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type)
    : query_type(query_type), noise_perc(""), dataset_name(dataset_name) {
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path)
    : VectorDataLoader(dataset_name, query_type) {
    this->directory_path = directory_path;
    this->noise_perc = noise_perc;
}

void VectorDataLoader::initializeDataMaps(){
    // SIFT10M: an instance from the DARTH paper
    baseVectorsMap["SIFT10M"] = directory_path + "SIFT10M/base.10M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.1M.fvecs";
    queryTypeToVectorsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
    
    queryTypeToGroundtruthsMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.groundtruth.1M.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT10M"] = directory_path + "SIFT10M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT10M"] = directory_path + "SIFT10M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["SIFT10M"] = directory_path + "SIFT10M/query.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT10M"] = directory_path + "SIFT10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    // deepimage
    baseVectorsMap["deepimage"] = directory_path + "deepimage/base.9.99M.fvecs";
    queryTypeToVectorsMap[TRAINING]["deepimage"] = directory_path + "deepimage/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["deepimage"] = directory_path + "deepimage/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["deepimage"] = directory_path + "deepimage/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["deepimage"] = directory_path + "deepimage/learn.groundtruth.10K.k100.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["deepimage"] = directory_path + "deepimage/validation.groundtruth.1K.k100.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["deepimage"] = directory_path + "deepimage/query.groundtruth.10K.k100.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["deepimage"] = directory_path + "deepimage/learn.groundtruth.10K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["deepimage"] = directory_path + "deepimage/validation.groundtruth.1K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["deepimage"] = directory_path + "deepimage/query.groundtruth.10K.k100.fvecs";

    // glove100    
    baseVectorsMap["glove100"] = directory_path + "glove100/base.1.18M.fvecs";
    queryTypeToVectorsMap[TRAINING]["glove100"] = directory_path + "glove100/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["glove100"] = directory_path + "glove100/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["glove100"] = directory_path + "glove100/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["glove100"] = directory_path + "glove100/learn.groundtruth.10K.k100.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["glove100"] = directory_path + "glove100/validation.groundtruth.1K.k100.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["glove100"] = directory_path + "glove100/query.groundtruth.10K.k100.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["glove100"] = directory_path + "glove100/learn.groundtruth.10K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["glove100"] = directory_path + "glove100/validation.groundtruth.1K.k100.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["glove100"] = directory_path + "glove100/query.groundtruth.10K.k100.fvecs";

    // msmarco_v1
    baseVectorsMap["msmarco_v1"] = directory_path + "msmarco_v1/base.8.84M.fvecs";
    queryTypeToVectorsMap[TRAINING]["msmarco_v1"] = directory_path + "msmarco_v1/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["msmarco_v1"] = directory_path + "msmarco_v1/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["msmarco_v1"] = directory_path + "msmarco_v1/query.6.98K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["msmarco_v1"] = directory_path + "msmarco_v1/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["msmarco_v1"] = directory_path + "msmarco_v1/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["msmarco_v1"] = directory_path + "msmarco_v1/query.groundtruth.6.98K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["msmarco_v1"] = directory_path + "msmarco_v1/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["msmarco_v1"] = directory_path + "msmarco_v1/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["msmarco_v1"] = directory_path + "msmarco_v1/query.groundtruth.6.98K.k1000.fvecs";

    // msmarco_v2.1    
    baseVectorsMap["msmarco_v2.1"] = directory_path + "msmarco_v2.1/base.18.38M.fvecs";
    queryTypeToVectorsMap[TRAINING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/query.1.67K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/query.groundtruth.1.67K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["msmarco_v2.1"] = directory_path + "msmarco_v2.1/query.groundtruth.1.67K.k1000.fvecs";

    // laion_i2i    
    baseVectorsMap["laion_i2i"] = directory_path + "laion_i2i/base.30.65M.fvecs";

    queryTypeToVectorsMap[TRAINING]["laion_i2i"] = directory_path + "laion_i2i/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["laion_i2i"] = directory_path + "laion_i2i/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["laion_i2i"] = directory_path + "laion_i2i/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["laion_i2i"] = directory_path + "laion_i2i/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["laion_i2i"] = directory_path + "laion_i2i/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["laion_i2i"] = directory_path + "laion_i2i/query.groundtruth.10K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["laion_i2i"] = directory_path + "laion_i2i/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["laion_i2i"] = directory_path + "laion_i2i/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["laion_i2i"] = directory_path + "laion_i2i/query.groundtruth.10K.k1000.fvecs";

    // laion_t2i
    baseVectorsMap["laion_t2i"] = directory_path + "laion_t2i/base.30.65M.fvecs";

    queryTypeToVectorsMap[TRAINING]["laion_t2i"] = directory_path + "laion_t2i/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["laion_t2i"] = directory_path + "laion_t2i/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["laion_t2i"] = directory_path + "laion_t2i/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["laion_t2i"] = directory_path + "laion_t2i/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["laion_t2i"] = directory_path + "laion_t2i/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["laion_t2i"] = directory_path + "laion_t2i/query.groundtruth.10K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["laion_t2i"] = directory_path + "laion_t2i/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["laion_t2i"] = directory_path + "laion_t2i/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["laion_t2i"] = directory_path + "laion_t2i/query.groundtruth.10K.k1000.fvecs";

    // uniform_cluster
    baseVectorsMap["uniform_cluster"] = directory_path + "uniform_cluster/base.10M.fvecs";

    queryTypeToVectorsMap[TRAINING]["uniform_cluster"] = directory_path + "uniform_cluster/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["uniform_cluster"] = directory_path + "uniform_cluster/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["uniform_cluster"] = directory_path + "uniform_cluster/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["uniform_cluster"] = directory_path + "uniform_cluster/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["uniform_cluster"] = directory_path + "uniform_cluster/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["uniform_cluster"] = directory_path + "uniform_cluster/query.groundtruth.10K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["uniform_cluster"] = directory_path + "uniform_cluster/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["uniform_cluster"] = directory_path + "uniform_cluster/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["uniform_cluster"] = directory_path + "uniform_cluster/query.groundtruth.10K.k1000.fvecs";

    // zipfian_cluster
    baseVectorsMap["zipfian_cluster"] = directory_path + "zipfian_cluster/base.10M.fvecs";

    queryTypeToVectorsMap[TRAINING]["zipfian_cluster"] = directory_path + "zipfian_cluster/learn.10K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["zipfian_cluster"] = directory_path + "zipfian_cluster/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["zipfian_cluster"] = directory_path + "zipfian_cluster/query.10K.fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["zipfian_cluster"] = directory_path + "zipfian_cluster/learn.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["zipfian_cluster"] = directory_path + "zipfian_cluster/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["zipfian_cluster"] = directory_path + "zipfian_cluster/query.groundtruth.10K.k1000.ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["zipfian_cluster"] = directory_path + "zipfian_cluster/learn.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["zipfian_cluster"] = directory_path + "zipfian_cluster/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["zipfian_cluster"] = directory_path + "zipfian_cluster/query.groundtruth.10K.k1000.fvecs";
}

float* VectorDataLoader::loadDB(size_t* d_out, size_t* n_out) {
    std::string baseVectorsPath = baseVectorsMap[dataset_name];
    
    printf(">> Loading base vectors from: %s\n", baseVectorsPath.c_str());
    float *db = fvecs_read(baseVectorsPath.c_str(), d_out, n_out);

    return db;
}

float* VectorDataLoader::loadQueries(size_t* d_out, size_t* n_out) {
    std::string queryVectorsPath = queryTypeToVectorsMap[query_type][dataset_name];
    
    printf(">> Loading queries from: %s\n", queryVectorsPath.c_str());
    float *queries = fvecs_read(queryVectorsPath.c_str(), d_out, n_out);

    // skip first 5000 queries:
    //*n_out -= 5000;
    //queries += 5000 * *d_out;

    return queries;
}

int* VectorDataLoader::loadQueriesGroundtruths(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthsPath = queryTypeToGroundtruthsMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruths from: %s\n", queryGroundtruthsPath.c_str());
    int *gt = ivecs_read(queryGroundtruthsPath.c_str(), k_out, n_out);

    return gt;
}

float* VectorDataLoader::loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthDistancesPath = queryTypeToGroundtruthDistancesMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruth distances from: %s\n", queryGroundtruthDistancesPath.c_str());
    float *gt = fvecs_read(queryGroundtruthDistancesPath.c_str(), k_out, n_out);

    return gt;
}


