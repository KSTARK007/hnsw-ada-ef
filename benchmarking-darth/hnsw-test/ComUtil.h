#pragma once

#include <faiss/Index.h>

struct RecallStats {
    double avg;
    double p1;
    double p5;
};

RecallStats recall_at_k_stats(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose = false);


void get_queries(
        float* vecsQ,
        faiss::idx_t nQ,
        int dQ,
        faiss::idx_t* gt,
        float *gt_dist,
        faiss::idx_t k,
        faiss::idx_t sample_size,
        float** vecsQ_out,
        faiss::idx_t** indicesQ_out,
        faiss::idx_t** gtQ_out,
        float** gt_dist_out,
        bool shuffle = false);

double elapsed();

