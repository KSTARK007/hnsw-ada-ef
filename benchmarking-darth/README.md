
# Benchmarking DARTH and LAET Quickstart
This repository is built based on the DARTH repository (https://github.com/MChatzakis/DARTH), and it provides the code for reproducing the experimental results of DARTH, LAET, and HNSW (FAISS), which are reported in the Ada-ef paper. Ensure the Ada-ef experiments are set up beforehand, as these scripts load datasets from the Ada-ef root directory. 

## Requirements
- C++17-capable compiler (validated with `gcc 12.3.0`)
- CMake ≥ 3.26
- LightGBM 4.6.0.99

## Environment Variables
- `DARTH_ROOT`: absolute path to this repository， `export DARTH_ROOT=$ADA_EF_ROOT/benchmarking-darth`
- `LIGHTGBM_ROOT`: installation path of LightGBM (`lightgbm-install`), `export LIGHTGBM_ROOT=/path/to/lightgbm-install`

## Build FAISS with DARTH
```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_OPT_LEVEL=avx512 \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3" \
  -DCMAKE_INSTALL_PREFIX=$LIGHTGBM_ROOT \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -C build -j faiss
make -C build -j hnsw_test
```
## Python Dependencies
```bash
pip install -r requirements.txt
```

## Experiment Workflow
1. Preprocess datasets with `python notebooks_scripts/utils/organize_datasets.py`.
2. Build HNSW indexes via `experiments/hnsw_create_index.sh`.
3. Generate training data with `experiments/hnsw_training_data_generation.sh`.
4. Train models: `python notebooks_scripts/predictor_training.py` for DARTH, `python notebooks_scripts/laet_training.py` for LAET; use the extracted statistics to configure parameters.
5. Evaluate learned models using `experiments/hnsw_darth_test.sh` or `bash experiments/hnsw_laet_test.sh`.
6. Run the FAISS HNSW baseline with `experiments/hnsw_no_early_stop_testing2.sh`. 
7. Measure per-query latency via `experiments/hnsw_darth_test_per_query.sh` and `bash experiments/hnsw_laet_test_per_query.sh`.