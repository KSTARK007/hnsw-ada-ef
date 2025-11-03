#include "ComUtil.h"

#include <unordered_set>
#include <cstdio>
#include <sys/time.h>
#include <vector>
#include <algorithm>

RecallStats recall_at_k_stats(
        const faiss::idx_t* gt,
        const faiss::idx_t* retrieved,
        const float* distances,
        int k,
        int nq,
        const faiss::idx_t* gt_all,
        int k_all,
        bool verbose) {
    double total = 0.0;
    std::vector<double> recalls;
    recalls.reserve(nq);
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<faiss::idx_t> gt_set(gt + i * k, gt + (i + 1) * k);
        int matches = 0;

        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i * k + j])) {
                matches++;
            }
        }

        double recall = (double)matches / (double)k;
        recalls.push_back(recall);

        if (verbose) {
            printf("Query[%d]: Recall@%d: %.4f. NN: %f NN_id: %ld\n",
                   i,
                   k,
                   recall,
                   distances[i * k],
                   retrieved[i * k]);

            if (gt_all[i * k_all] != retrieved[i * k]) {
                for (int j = 0; j < k_all; j++) {
                    if (gt_all[i * k_all + j] == retrieved[i * k]) {
                        printf("    NN_id found at %d-th position.\n", j + 1);
                        break;
                    }
                }
            }
        }

        total += recall;
    }

    double avg = total / (double)(nq);
    std::sort(recalls.begin(), recalls.end());

    auto percentile = [&](double p) -> double {
        if (recalls.empty()) return 0.0;
        double rank = p * (recalls.size() - 1);
        size_t lower = static_cast<size_t>(rank);
        size_t upper = std::min(lower + 1, recalls.size() - 1);
        double frac = rank - lower;
        return recalls[lower] * (1.0 - frac) + recalls[upper] * frac;
    };

    double p1 = percentile(0.01);
    double p5 = percentile(0.05);


    return {avg, p1, p5};
}

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
        bool shuffle) {
    float* vecsQ_sample = new float[sample_size * dQ];

    faiss::idx_t* gtQ_sample = new faiss::idx_t[sample_size * k];
    float* gt_dist_sample = new float[sample_size * k];

    faiss::idx_t* indicesQ_sample = new faiss::idx_t[sample_size];

    faiss::idx_t* all_indices = new faiss::idx_t[nQ];
    for (faiss::idx_t i = 0; i < nQ; i++) {
        all_indices[i] = i;
    }

    // if (shuffle) {
    //     std::shuffle(
    //             all_indices,
    //             all_indices + nQ,
    //             std::default_random_engine(SEED));
    // }

    for (faiss::idx_t i = 0; i < sample_size; i++) {
        int idx = all_indices[i];

        for (faiss::idx_t j = 0; j < dQ; j++) {
            vecsQ_sample[i * dQ + j] = vecsQ[idx * dQ + j];
        }

        for (faiss::idx_t j = 0; j < k; j++) {
            gtQ_sample[i * k + j] = gt[idx * k + j];
            gt_dist_sample[i * k + j] = gt_dist[idx * k + j];
        }

        indicesQ_sample[i] = idx;
    }
    
    *vecsQ_out = vecsQ_sample;
    *indicesQ_out = indicesQ_sample;
    *gtQ_out = gtQ_sample;
    *gt_dist_out = gt_dist_sample;

    return;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}


