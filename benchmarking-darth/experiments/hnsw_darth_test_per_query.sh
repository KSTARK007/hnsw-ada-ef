#!/usr/bin/env bash
# cd ../
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
RESULTS_DIRECTORY=$DARTH_ROOT/experiments/results
MODEL_DIRECTORY=$DARTH_ROOT/predictor_models
DATASET_DIRECTORY=$DARTH_ROOT/datasets/processed/

mode=early-stop-testing
n_estim=100
train_queries=10000
experiment_times=1

# ds M efC efS li k sample target_recall initial_prediction_interval min_prediction_interval # params are based on stats obtained via extrac_data_data_stats.py
dataset_params=(  
  "deepimage      16 500 500  2 100  10000 0.95 1595 319"
  "msmarco_v2.1   16 500 2000 2 1000 1677  0.95 4542 908"
)

mkdir -p ${RESULTS_DIRECTORY}/${mode}

for dataset_param in "${dataset_params[@]}"
do
    read ds M efC efS li k sample target_recall initial_prediction_interval min_prediction_interval <<< "$dataset_param"

    echo ""
    echo "--------------- ${ds} ---------------"
    echo ""

    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}

    echo ""
    echo "--------------- k=${k} ---------------"
    echo ""
    
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/detailed/
    mkdir ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/noisy

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
        --initial-prediction-interval ${initial_prediction_interval} \
        --min-prediction-interval ${min_prediction_interval} \
        --query-type testing \
        --predictor-model-path ${MODEL_DIRECTORY}/darth/${ds}_M${M}_efC${efC}_efS${efS}_s${train_queries}_k${k}_nestim${n_estim}_li${li}_all_feats.txt \
        --output $RESULTS_DIRECTORY/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}_t${time}.txt
    done
    # python result_merger.py \
    #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/times/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval} \
    #       ${RESULTS_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${sample}_tr${target_recall}_ipi${initial_prediction_interval}_mpi${min_prediction_interval}.txt \
    #       $experiment_times
    echo ""
    echo ""
done