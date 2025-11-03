#!/usr/bin/env bash
# cd ..
# cmake -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -B build -S .
# make -C build -j faiss
# make -C build -j hnsw_test
# cd experiments

echo ""
echo ""
echo "============================="
echo ""
echo ""

INDEX_DIRECTORY=$DARTH_ROOT/hnsw-index
DATASET_DIRECTORY=$DARTH_ROOT/datasets/processed/
RESULTS_DIRECTORY=$DARTH_ROOT/experiments/results
MODEL_DIRECTORY=$DARTH_ROOT/predictor_models

train_queries=10000
experiment_times=1

mode=laet-early-stop-testing
mkdir -p ${RESULTS_DIRECTORY}/${mode}

# ds M efC efS li k sample target_recall F multiplier
dataset_params=(
  "deepimage      16 500 500  2 100  10000 0.95 450    1.0"
  "msmarco_v2.1   16 500 2000 2 1000 1677  0.95 1704   1.0"
)

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample target_recall F multiplier <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/noisy

    echo ""
    echo "--------------- k=${k} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy/times/
    
    for time in $(seq 1 $experiment_times)
    do
        ./../build/hnsw-test/hnsw_test \
                --dataset ${ds} \
                --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                --query-num ${sample} --k $k \
                --mode ${mode} \
                --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                --dataset-dir-prefix ${DATASET_DIRECTORY} \
                --target-recall ${target_recall} \
                --fixed-amount-of-search ${F} --prediction-multiplier ${multiplier} \
                --query-type testing \
                --predictor-model-path ${MODEL_DIRECTORY}/laet/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}.txt \
                --output ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_t${time}.txt 

    done
  #   python result_merger.py \
  #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall} \
  #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}.txt \
  #       $experiment_times
    echo ""
    echo ""
done
