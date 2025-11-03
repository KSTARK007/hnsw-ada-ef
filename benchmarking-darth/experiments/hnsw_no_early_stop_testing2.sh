#!/usr/bin/env bash

# $HOME/cmake-4.1.2-linux-x86_64/bin/cmake -B build -S . \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DFAISS_ENABLE_GPU=OFF \
#  -DFAISS_OPT_LEVEL=avx512 \
#  -DBUILD_SHARED_LIBS=ON  \
#  -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
#  -DCMAKE_INSTALL_PREFIX=$HOME/lightgbm-install \
#  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#  -DCMAKE_CXX_FLAGS_RELEASE="-O3"
# make -C build -j faiss
# make -C build -j hnsw_test

echo ""
echo ""
echo "============================="
echo ""
echo ""

dataset_params=(
    "deepimage       16 500 500  100  10000"
    "glove100        16 500 500  100  10000"
    "msmarco_v1      16 500 500  1000 6980"
    "msmarco_v2.1    16 500 2000 1000 1677"
    "laion_i2i       16 500 2000 1000 10000"
    "laion_t2i       16 500 2000 1000 10000"
    "uniform_cluster 16 500 2000 1000 10000"
    "zipfian_cluster 16 500 2000 1000 10000"
    )

INDEX_DIRECTORY=$DARTH_ROOT/hnsw-index
DATASET_DIRECTORY=$DARTH_ROOT/datasets/processed/

mode=no-early-stop-testing-2
query_type=testing
for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS k sample<<< "$dataset_param"

    ../build/hnsw-test/hnsw_test \
        --dataset ${ds} \
        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
        --query-num ${sample} --k ${k} \
        --mode ${mode} \
        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
        --query-type ${query_type} \
        --dataset-dir-prefix ${DATASET_DIRECTORY}
done