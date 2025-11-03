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

# Each entry is: dataset M efC efS k
dataset_params=(
    "deepimage      16 500 500 100"
    "glove100       16 500 500 100"
    "msmarco_v1     16 500 2000 1000"
    "msmarco_v2.1   16 500 2000 1000"
    "uniform_cluster 16 500 2000 1000"
    "zipfian_cluster 16 500 2000 1000"
    "laion_i2i       16 500 2000 1000"
    "laion_t2i       16 500 2000 1000"
)


TRAINING_DATA_DIRECTORY=$DARTH_ROOT/et_training_data
INDEX_DIRECTORY=$DARTH_ROOT/hnsw-index
DATASET_DIRECTORY=$DARTH_ROOT/datasets/processed/

mode=early-stop-training
mkdir ${TRAINING_DATA_DIRECTORY}
mkdir ${TRAINING_DATA_DIRECTORY}/${mode}
for query_num in 10000
do
    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS k<<< "$dataset_param"

        mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}

        echo ""
        echo "--------------- ${ds} ---------------"
        echo ""

        # for k in ${k_values[@]}
        # do
            mkdir ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}

            for logging_interval in 2
            do
                ../build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${query_num} --k $k \
                        --mode ${mode} \
                        --logging-interval ${logging_interval} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --query-type training \
                        --output ${TRAINING_DATA_DIRECTORY}/${mode}/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt
            done
        # done

        echo ""
        echo ""
    done
done

echo "Training data generation completed."

exit

# Generate the validation data
# We need the detailed data with the target variables to evaluate the model on the val dataset for 1000 queries
mode=early-stop-training
mkdir ./results/validation_logging/
for query_num in 1000
do
    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS k<<< "$dataset_param"

        mkdir ./results/validation_logging/${ds}

        echo ""
        echo "--------------- ${ds} ---------------"
        echo ""

        # for k in ${k_values[@]}
        # do
            mkdir ./results/validation_logging/${ds}/k${k}

            for logging_interval in 1
            do
                ../build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${query_num} --k $k \
                        --output ./results/validation_logging/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt \
                        --mode ${mode} \
                        --logging-interval ${logging_interval} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --query-type validation 
            done
        # done

        echo ""
        echo ""
    done
done


# We do the same for the testing data
# We need this to evaluate the final test performance of the model
mode=early-stop-training
mkdir ./results/test_logging/
for query_num in 1000
do
    for dataset_param in "${dataset_params[@]}"
    do
        read ds M efC efS k<<< "$dataset_param"

        mkdir ./results/test_logging/${ds}

        echo ""
        echo "--------------- ${ds} ---------------"
        echo ""

        # for k in ${k_values[@]}
        # do
            mkdir ./results/test_logging/${ds}/k${k}

            for logging_interval in 1
            do
                ../build/hnsw-test/hnsw_test \
                        --dataset ${ds} \
                        --M ${M} --efConstruction ${efC} --efSearch ${efS} \
                        --query-num ${query_num} --k $k \
                        --output ./results/test_logging/${ds}/k${k}/M${M}_efC${efC}_efS${efS}_qs${query_num}_li${logging_interval}.txt \
                        --mode ${mode} \
                        --logging-interval ${logging_interval} \
                        --index-filepath ${INDEX_DIRECTORY}/${ds}/${ds}.M${M}.efC${efC}.index \
                        --dataset-dir-prefix ${DATASET_DIRECTORY} \
                        --query-type testing 
            done
        # done

        echo ""
        echo ""
    done
done


exit 




