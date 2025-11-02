#pragma once

#include <H5Cpp.h>
#include "../hnswlib/adaptive_ef.h"

void load_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors,
               hnswdis::MatrixXi &neighbors)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read "train" dataset
    // --------------------
    {
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();

        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims, nullptr);

        data_vectors.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        train_dataset.read(data_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read "test" dataset
    // --------------------
    {
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        query_vectors.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        test_dataset.read(query_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 3. Read "neighbors" dataset (int)
    // --------------------
    {
        H5::DataSet neighbors_dataset = file.openDataSet("neighbors");
        H5::DataSpace neighbors_dataspace = neighbors_dataset.getSpace();

        hsize_t neighbors_dims[2];
        neighbors_dataspace.getSimpleExtentDims(neighbors_dims, nullptr);

        neighbors.resize(neighbors_dims[0], neighbors_dims[1]);

        // Read raw data into row-major
        neighbors_dataset.read(neighbors.data(), H5::PredType::NATIVE_INT);
    }
}

void load_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read "train" dataset
    // --------------------
    {
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();

        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims, nullptr);

        data_vectors.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        train_dataset.read(data_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read "test" dataset
    // --------------------
    {
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        query_vectors.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        test_dataset.read(query_vectors.data(), H5::PredType::NATIVE_FLOAT);
    }
}

void load_hdf5(const std::string &path,
               const std::string &first_name,
               hnswdis::MatrixXf &first_matrix,
               const std::string &second_name,
               hnswdis::MatrixXi &second_matrix)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read dataset with name "first_name"
    // --------------------
    {
        H5::DataSet first_dataset = file.openDataSet(first_name);
        H5::DataSpace first_dataspace = first_dataset.getSpace();

        hsize_t train_dims[2];
        first_dataspace.getSimpleExtentDims(train_dims, nullptr);

        first_matrix.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        first_dataset.read(first_matrix.data(), H5::PredType::NATIVE_FLOAT);
    }

    // --------------------
    // 2. Read dataset with name "second_name"
    // --------------------
    {
        H5::DataSet second_dataset = file.openDataSet(second_name);
        H5::DataSpace test_dataspace = second_dataset.getSpace();

        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims, nullptr);

        second_matrix.resize(test_dims[0], test_dims[1]);
        // Read raw data into row-major
        second_dataset.read(second_matrix.data(), H5::PredType::NATIVE_INT);
    }
}

void load_hdf5(const std::string &path,
               const std::string &first_name,
               hnswdis::MatrixXf &first_matrix)
{

    // Open HDF5 file in read-only mode
    H5::H5File file(path, H5F_ACC_RDONLY);

    // --------------------
    // 1. Read dataset with name "first_name"
    // --------------------
    {
        H5::DataSet first_dataset = file.openDataSet(first_name);
        H5::DataSpace first_dataspace = first_dataset.getSpace();

        hsize_t train_dims[2];
        first_dataspace.getSimpleExtentDims(train_dims, nullptr);

        first_matrix.resize(train_dims[0], train_dims[1]);
        // Read from HDF5 (row-major ordering)
        first_dataset.read(first_matrix.data(), H5::PredType::NATIVE_FLOAT);
    }
}

void save_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXf &data_vectors,
               hnswdis::MatrixXi &neighbors)
{
    // Create an HDF5 file
    H5::H5File file(path, H5F_ACC_TRUNC);

    float *test_data = query_vectors.data();
    float *train_data = data_vectors.data();
    int *neighbors_data = neighbors.data();

    hsize_t query_dims[2] = {query_vectors.rows(), query_vectors.cols()};
    H5::DataSpace query_dataspace(2, query_dims);

    hsize_t data_dims[2] = {data_vectors.rows(), data_vectors.cols()};
    H5::DataSpace data_dataspace(2, data_dims);

    hsize_t neighbors_dims[2] = {neighbors.rows(), neighbors.cols()};
    H5::DataSpace neighbors_dataspace(2, neighbors_dims);

    // Create "test" dataset and write data
    H5::DataSet dataset_test = file.createDataSet("test", H5::PredType::NATIVE_FLOAT, query_dataspace);
    dataset_test.write(test_data, H5::PredType::NATIVE_FLOAT);

    // Create "train" dataset and write data
    H5::DataSet dataset_train = file.createDataSet("train", H5::PredType::NATIVE_FLOAT, data_dataspace);
    dataset_train.write(train_data, H5::PredType::NATIVE_FLOAT);

    // Create "neighbors" dataset and write data
    H5::DataSet dataset_neighbors = file.createDataSet("neighbors", H5::PredType::NATIVE_INT, neighbors_dataspace);
    dataset_neighbors.write(neighbors_data, H5::PredType::NATIVE_INT);

    std::cout << "HDF5 file created with Eigen matrices successfully!" << std::endl;
}

void save_hdf5(const std::string &path,
               hnswdis::MatrixXf &query_vectors,
               hnswdis::MatrixXi &neighbors)
{
    // Create an HDF5 file
    H5::H5File file(path, H5F_ACC_TRUNC);

    float *test_data = query_vectors.data();
    int *neighbors_data = neighbors.data();

    hsize_t query_dims[2] = {query_vectors.rows(), query_vectors.cols()};
    H5::DataSpace query_dataspace(2, query_dims);

    hsize_t neighbors_dims[2] = {neighbors.rows(), neighbors.cols()};
    H5::DataSpace neighbors_dataspace(2, neighbors_dims);

    // Create "test" dataset and write data
    H5::DataSet dataset_test = file.createDataSet("test", H5::PredType::NATIVE_FLOAT, query_dataspace);
    dataset_test.write(test_data, H5::PredType::NATIVE_FLOAT);

    // Create "neighbors" dataset and write data
    H5::DataSet dataset_neighbors = file.createDataSet("neighbors", H5::PredType::NATIVE_INT, neighbors_dataspace);
    dataset_neighbors.write(neighbors_data, H5::PredType::NATIVE_INT);

    std::cout << "HDF5 file created with Eigen matrices successfully!" << std::endl;
}

void normalize_matrix(hnswdis::MatrixXf &matrix)
{
    for (int i = 0; i < matrix.rows(); ++i)
    {
        float norm = matrix.row(i).norm();
        norm = 1.0f / (norm + 1e-30f);
        matrix.row(i) *= norm;
    }
}

void compute_and_save_gound_truth(const std::string &query_data_path, const std::string &query_data_neighbour_path, const std::string &metric, const int k, bool save = true)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto data_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();

    load_hdf5(query_data_path, *query_vectors_ptr, *data_vectors_ptr);

    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(*data_vectors_ptr);
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Data vectors dimensions: " << data_vectors_ptr->rows() << " x " << data_vectors_ptr->cols() << std::endl;
    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;

    auto ground_truth = hnswdis::compute_ground_truth(*query_vectors_ptr, *data_vectors_ptr, metric, k);

    if (save)
    {
        std::cout << "Saving ground truth to " << query_data_neighbour_path << std::endl;
        save_hdf5(query_data_neighbour_path, *query_vectors_ptr, *data_vectors_ptr, ground_truth);
    }
}

void build_index(
    const std::string &hdf5_path,
    const std::string &index_path,
    const int M,
    const int ef_construction,
    const std::string &metric,
    const int num_threads)
{
    hnswdis::MatrixXf query_vectors, data_vectors;
    hnswdis::MatrixXi neighbors;
    // Load the data
    load_hdf5(hdf5_path, query_vectors, data_vectors, neighbors);
    std::cout << "[Query vectors] rows: " << query_vectors.rows()
              << ", cols: " << query_vectors.cols() << std::endl;

    std::cout << "[Data vectors]  rows: " << data_vectors.rows()
              << ", cols: " << data_vectors.cols() << std::endl;

    std::cout << "[Neighbors] rows: " << neighbors.rows()
              << ", cols: " << neighbors.cols() << std::endl;
    const int dim = data_vectors.cols();          // Dimension of the elements
    const int max_elements = data_vectors.rows(); // Maximum number of elements, should be known beforehand

    if (metric == "cd")
    {
        // normalize the data vectors
        normalize_matrix(data_vectors);
    }

    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, dim);
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), max_elements, M, ef_construction);

    auto start = std::chrono::high_resolution_clock::now();
    hnswdis::ParallelFor(0, max_elements, num_threads, [&](size_t row_id, size_t threadId)
                         { alg_hnsw->addPoint((void *)(data_vectors.row(row_id).data()), row_id); });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index built in " << duration.count() << " ms" << std::endl;

    // Query the fist 1k elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < 1000; ++i)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(
            data_vectors.row(i).data(), 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    std::cout << "First 1k Data Points Recall: " << correct / 1000 << "\n";

    // Query the last 1k elements for themselves and measure recall
    correct = 0;
    for (int i = max_elements - 1000; i < max_elements; ++i)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(
            data_vectors.row(i).data(), 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    std::cout << "Last 1k Data Points Recall: " << correct / 1000 << "\n";

    alg_hnsw->saveIndex(index_path);
}

std::tuple<
    std::shared_ptr<hnswlib::HierarchicalNSW<float>>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXf>,
    std::shared_ptr<hnswdis::MatrixXi>,
    std::shared_ptr<hnswlib::SpaceInterface<float>>>
load_index_and_data(const std::string &hdf5_path, const std::string &index_path, const std::string &metric)
{
    auto query_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto data_vectors_ptr = std::make_shared<hnswdis::MatrixXf>();
    auto neighbors_ptr = std::make_shared<hnswdis::MatrixXi>();

    // Load the data
    load_hdf5(hdf5_path, *query_vectors_ptr, *data_vectors_ptr, *neighbors_ptr);

    if (metric == "cd")
    {
        std::cout << "Normalize the data vectors" << std::endl;
        normalize_matrix(*data_vectors_ptr);
        normalize_matrix(*query_vectors_ptr);
    }

    std::cout << "Data vectors dimensions: " << data_vectors_ptr->rows() << " x " << data_vectors_ptr->cols() << std::endl;
    std::cout << "Query vectors dimensions: " << query_vectors_ptr->rows() << " x " << query_vectors_ptr->cols() << std::endl;
    std::cout << "Neighbors dimensions: " << neighbors_ptr->rows() << " x " << neighbors_ptr->cols() << std::endl;

    std::shared_ptr<hnswlib::SpaceInterface<float>> space = hnswdis::init_space(metric, query_vectors_ptr->cols());

    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), index_path);

    std::cout << "Index loaded" << std::endl;

    std::cout << "Dimension of space:" << *(size_t *)(space->get_dist_func_param()) << std::endl;
    std::cout << "Data size of space:" << space->get_data_size() << std::endl;

    return std::make_tuple(alg_hnsw, query_vectors_ptr, data_vectors_ptr, neighbors_ptr, space);
}

void print_score_distribution(const std::vector<float> &score_list)
{
    std::vector<int> score_ranges(10, 0);

    for (const auto &score : score_list)
    {
        int range_index = static_cast<int>(score / 10);
        if (range_index >= 0 && range_index < 10)
        {
            ++score_ranges[range_index];
        }
        else if (range_index == 10)
        {
            ++score_ranges[9];
        }
        else
        {
            std::cerr << "Invalid score: " << score << std::endl;
        }
    }

    for (int i = 0; i < score_ranges.size(); ++i)
    {
        std::cout << "[" << i * 10 << "," << (i + 1) * 10 << "]: " << score_ranges[i] << std::endl;
    }
}

void search_and_score(
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const std::string &metric,
    const size_t k,
    const size_t statics_length,
    const float quantile_step)
{

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < query_vectors.rows(); i++)
    {
        auto ret = alg_hnsw.searchKnn(
            query_vectors.row(i).data(), k);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Search time: " << duration.count() << " ms" << std::endl;

    std::shared_ptr<hnswdis::Estimator> estimator = hnswdis::init_estimator(metric, data_vectors);
    hnswdis::ApproximatedScoreCalculator score_cal(estimator, quantile_step);
    std::vector<float> score_list;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < query_vectors.rows(); i++)
    {
        auto ret = alg_hnsw.adaptiveSearchKnn(
            query_vectors.row(i).data(), k, statics_length, score_cal);
        score_list.push_back(ret.second);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Search time: " << duration.count() << " ms" << std::endl;
    print_score_distribution(score_list);
}

void print_recall_distribution(std::vector<float> &recalls)
{
    std::vector<int> recall_distribution(101, 0);
    for (const auto &recall : recalls)
    {
        int range_index = static_cast<int>(recall * 100);
        if (range_index >= 0 && range_index <= 100)
        {
            ++recall_distribution[range_index];
        }
        else
        {
            std::cerr << "Invalid recall value: " << recall << std::endl;
        }
    }

    for (int i = 0; i < recall_distribution.size(); ++i)
    {
        if (recall_distribution[i] == 0)
            continue;
        std::cout << "[" << i * 0.01 << "," << (i + 1) * 0.01 << "]: " << recall_distribution[i] << std::endl;
    }
}

void adaptive_search(
    const std::string &dataset,
    const int repeat,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length,
    const float expected_recall)
{
    std::vector<int64_t> time;
    time.reserve(repeat);

    std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
        std::make_tuple(statics_length, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

    for (int i = 0; i < repeat; ++i)
    {
        std::vector<std::vector<size_t>> result;
        for (int j = 0; j < query_vectors.rows(); ++j)
        {
            result.push_back(std::vector<size_t>(k, 0));
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < query_vectors.rows(); ++j)
        {
            auto pq = alg_hnsw.adaptiveSearchKnnTest(
                query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);
            std::vector<size_t> &labels = result[j];
            {
                size_t count = pq.size();
                while (!pq.empty())
                {
                    labels[--count] = pq.top().second;
                    pq.pop();
                }
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Search time: " << duration.count() << " ms" << std::endl;

        time.push_back(duration.count());

        auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
        // std::cout << "Average recall: " << std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size() << std::endl;

        // print_recall_distribution(recalls);

        if (i == repeat - 1)
        {
            // auto recalls = hnswdis::compute_recall(ground_truth, result, false);
            // std::cout << "Average recall: " << std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size() << std::endl;

            auto avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
            std::cout << "Average Recall: " << avg_recall << std::endl;

            // print_recall_distribution(recalls);
            std::sort(recalls.begin(), recalls.end());
            size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
            size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
            float percentile_5 = recalls[index_5];
            float percentile_1 = recalls[index_1];
            std::cout << "5th percentile recall: " << percentile_5 << std::endl;
            std::cout << "1st percentile recall: " << percentile_1 << std::endl;

            int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
            for (const auto &recall : recalls)
            {
                if (recall >= 0.99)
                    count_high_recall_99++;
                if (recall >= 0.95)
                    count_high_recall_95++;
                if (recall >= 0.90)
                    count_high_recall_90++;
            }

            int num_queries = recalls.size();

            std::get<2>(exp_record) = avg_recall;
            std::get<3>(exp_record) = percentile_5;
            std::get<4>(exp_record) = percentile_1;

            std::get<5>(exp_record) = count_high_recall_99;
            std::get<6>(exp_record) = count_high_recall_95;
            std::get<7>(exp_record) = count_high_recall_90;
        }
    }
    std::sort(time.begin(), time.end());
    int64_t median_time = time[time.size() / 2];
    std::cout << "Median search time: " << median_time << " ms" << std::endl;
    std::cout << "Search times: ";
    for (const auto &t : time)
    {
        std::cout << t << " ms, ";
    }
    std::cout << std::endl;

    std::get<1>(exp_record) = median_time;

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "statisc_length, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;

    std::cout << std::get<0>(exp_record) << ", "
              << std::get<1>(exp_record) << ", "
              << std::get<2>(exp_record) << ", "
              << std::get<3>(exp_record) << ", "
              << std::get<4>(exp_record) << ", "
              << std::get<5>(exp_record) << ", "
              << std::get<6>(exp_record) << ", "
              << std::get<7>(exp_record) << std::endl;

    std::cout << "Experiment finished" << std::endl;
}

void adaptive_search_per_query_result(
    const std::string &dataset,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXf &data_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length,
    const float expected_recall)
{
    const int num_queries = query_vectors.rows();
    std::vector<std::vector<size_t>> result(num_queries, std::vector<size_t>(k, 0));

    std::vector<int64_t> latencies_ns(num_queries);
    std::vector<float> recalls(num_queries);

    for (int j = 0; j < num_queries; ++j)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto pq = alg_hnsw.adaptiveSearchKnnTest(
            query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);

        auto end = std::chrono::high_resolution_clock::now();
        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        latencies_ns[j] = latency_ns;

        // Extract top-k results
        size_t count = pq.size();
        while (!pq.empty())
        {
            result[j][--count] = pq.top().second;
            pq.pop();
        }

        // Compute recall for this query
        int correct = 0;
        for (size_t id : result[j])
        {
            for (int gt_idx = 0; gt_idx < k; ++gt_idx)
            {
                if (id == ground_truth(j, gt_idx))
                {
                    correct++;
                    break;
                }
            }
        }
        recalls[j] = static_cast<float>(correct) / k;
    }

    // === Summary statistics ===
    double avg_latency = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0) / num_queries;
    double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / num_queries;

    std::cout << "=== Per-Query Latency and Recall Results for " << dataset << " ===" << std::endl;
    std::cout << "QueryID, Latency(ns), Recall" << std::endl;
    for (int j = 0; j < num_queries; ++j)
    {
        std::cout << j << ", " << latencies_ns[j] << ", " << recalls[j] << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average Latency: " << avg_latency << " ns" << std::endl;
    std::cout << "Average Recall: " << avg_recall << std::endl;

    std::vector<int64_t> sorted_latencies = latencies_ns;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    std::cout << "95th percentile latency: " << sorted_latencies[(int)(num_queries * 0.95)] << " ns" << std::endl;
    std::cout << "99th percentile latency: " << sorted_latencies[(int)(num_queries * 0.99)] << " ns" << std::endl;

    std::cout << "Experiment finished." << std::endl;
}


void adaptive_ef_analysis(
    const std::string &dataset,
    const hnswlib::HierarchicalNSW<float> &alg_hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::ApproximatedScoreCalculator &score_cal,
    const size_t k,
    hnswdis::Sketch &sketch,
    const size_t statics_length)
{
    std::cout << "Adaptive EF Analysis for dataset: " << dataset << std::endl;
    for (int j = 0; j < query_vectors.rows(); ++j)
    {
        alg_hnsw.adaptiveSearchKnn(
            query_vectors.row(j).data(), k, statics_length, score_cal, &sketch);
    }
    std::cout << std::endl;
}

void baseline_search(
    const std::string &dataset,
    const int repeat,
    hnswlib::HierarchicalNSW<float> &hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const size_t k,
    const size_t ef_upper_bound)
{
    size_t ef = k;
    float avg_recall = 0.0f;
    // scheme for storing the results
    // ef, time, avg_recall, 5th percentile recall, 1st percentile recall, recall_above_99, recall_above_95, recall_above_90
    std::vector<std::tuple<size_t, size_t, float, float, float, int, int, int>> exp_results;
    while (exp_results.size() < 3 || avg_recall < 0.99)
    {
        std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
            std::make_tuple(ef, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

        std::cout << "ef: " << ef << std::endl;
        hnsw.setEf(ef);

        std::vector<int64_t> hnsw_search_time;
        hnsw_search_time.reserve(repeat);
        for (int i = 0; i < repeat; ++i)
        {
            std::vector<std::vector<size_t>> result;
            result.reserve(query_vectors.rows());
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < query_vectors.rows(); i++)
            {
                auto ret = hnsw.searchKnn(
                    query_vectors.row(i).data(), k); // transforming to clser first

                size_t count = ret.size();
                std::vector<size_t> labels(count);
                while (!ret.empty())
                {
                    labels[--count] = ret.top().second;
                    ret.pop();
                }

                result.push_back(std::move(labels));
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            hnsw_search_time.push_back(duration.count());

            if (i == repeat - 1)
            {
                auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
                avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
                std::cout << "Average Recall: " << avg_recall << std::endl;

                // print_recall_distribution(recalls);
                std::sort(recalls.begin(), recalls.end());
                size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
                size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
                float percentile_5 = recalls[index_5];
                float percentile_1 = recalls[index_1];
                std::cout << "5th percentile recall: " << percentile_5 << std::endl;
                std::cout << "1st percentile recall: " << percentile_1 << std::endl;

                int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
                for (const auto &recall : recalls)
                {
                    if (recall >= 0.99)
                        count_high_recall_99++;
                    if (recall >= 0.95)
                        count_high_recall_95++;
                    if (recall >= 0.90)
                        count_high_recall_90++;
                }

                int num_queries = recalls.size();

                std::get<0>(exp_record) = ef;

                std::get<2>(exp_record) = avg_recall;
                std::get<3>(exp_record) = percentile_5;
                std::get<4>(exp_record) = percentile_1;

                std::get<5>(exp_record) = count_high_recall_99;
                std::get<6>(exp_record) = count_high_recall_95;
                std::get<7>(exp_record) = count_high_recall_90;
            }
        }

        std::sort(hnsw_search_time.begin(), hnsw_search_time.end());
        int64_t median_time = hnsw_search_time[hnsw_search_time.size() / 2];
        std::cout << "Median search time: " << median_time << " ms" << std::endl;

        std::get<1>(exp_record) = median_time;
        exp_results.push_back(exp_record);

        std::cout << "Search times: ";
        for (const auto &t : hnsw_search_time)
        {
            std::cout << t << " ms, ";
        }
        std::cout << std::endl;

        if (ef > ef_upper_bound)
        {
            break;
        }

        if (ef >= 1600)
        {
            ef += 400;
        }
        else
        {
            ef *= 2;
        }
    }

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "ef, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;
    for (const auto &result : exp_results)
    {
        std::cout << std::get<0>(result) << ", "
                  << std::get<1>(result) << ", "
                  << std::get<2>(result) << ", "
                  << std::get<3>(result) << ", "
                  << std::get<4>(result) << ", "
                  << std::get<5>(result) << ", "
                  << std::get<6>(result) << ", "
                  << std::get<7>(result) << std::endl;
    }
    std::cout << "Experiment finished" << std::endl;
}

void search_with_patience_in_proximity(
    const std::string &dataset,
    const int repeat,
    hnswlib::HierarchicalNSW<float> &hnsw,
    const hnswdis::MatrixXf &query_vectors,
    const hnswdis::MatrixXi &ground_truth,
    const size_t k)
{

    std::cout << "Search with Patience in Proximity for dataset: " << dataset << std::endl;

    size_t ef = 3 * k;
    if (k >= 1000)
    {
        ef = 2 * k;
    }

    float avg_recall = 0.0f;
    // scheme for storing the results
    // ef, time, avg_recall, 5th percentile recall, 1st percentile recall, recall_above_99, recall_above_95, recall_above_90
    std::tuple<size_t, size_t, float, float, float, int, int, int> exp_record =
        std::make_tuple(ef, 0, 0.0f, 0.0f, 0.0f, 0, 0, 0);

    std::cout << "ef: " << ef << std::endl;
    hnsw.setEf(ef);

    std::vector<int64_t> hnsw_search_time;
    hnsw_search_time.reserve(repeat);
    for (int i = 0; i < repeat; ++i)
    {
        std::vector<std::vector<size_t>> result;
        result.reserve(query_vectors.rows());
        auto start_time = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < query_vectors.rows(); i++)
        {
            auto ret = hnsw.searchKnnWithPatienceInProximity(
                query_vectors.row(i).data(), k); // transforming to clser first

            size_t count = ret.size();
            std::vector<size_t> labels(count);
            while (!ret.empty())
            {
                labels[--count] = ret.top().second;
                ret.pop();
            }

            result.push_back(std::move(labels));
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        hnsw_search_time.push_back(duration.count());

        if (i == repeat - 1)
        {
            auto recalls = hnswdis::compute_recall(ground_truth, result, k, false);
            avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
            std::cout << "Average Recall: " << avg_recall << std::endl;

            // print_recall_distribution(recalls);
            std::sort(recalls.begin(), recalls.end());
            size_t index_5 = static_cast<size_t>(recalls.size() * 0.05);
            size_t index_1 = static_cast<size_t>(recalls.size() * 0.01);
            float percentile_5 = recalls[index_5];
            float percentile_1 = recalls[index_1];
            std::cout << "5th percentile recall: " << percentile_5 << std::endl;
            std::cout << "1st percentile recall: " << percentile_1 << std::endl;

            int count_high_recall_99 = 0, count_high_recall_95 = 0, count_high_recall_90 = 0;
            for (const auto &recall : recalls)
            {
                if (recall >= 0.99)
                    count_high_recall_99++;
                if (recall >= 0.95)
                    count_high_recall_95++;
                if (recall >= 0.90)
                    count_high_recall_90++;
            }

            int num_queries = recalls.size();

            std::get<0>(exp_record) = ef;

            std::get<2>(exp_record) = avg_recall;
            std::get<3>(exp_record) = percentile_5;
            std::get<4>(exp_record) = percentile_1;

            std::get<5>(exp_record) = count_high_recall_99;
            std::get<6>(exp_record) = count_high_recall_95;
            std::get<7>(exp_record) = count_high_recall_90;
        }
    }

    std::sort(hnsw_search_time.begin(), hnsw_search_time.end());
    int64_t median_time = hnsw_search_time[hnsw_search_time.size() / 2];
    std::cout << "Median search time: " << median_time << " ms" << std::endl;

    std::get<1>(exp_record) = median_time;

    std::cout << "Search times: ";
    for (const auto &t : hnsw_search_time)
    {
        std::cout << t << " ms, ";
    }
    std::cout << std::endl;

    std::cout << dataset << " experiment results:" << std::endl;
    std::cout << "ef, time, avg_recall, 5th_percentile_recall, 1st_percentile_recall, recall_above_99, recall_above_95, recall_above_90" << std::endl;

    std::cout << std::get<0>(exp_record) << ", "
              << std::get<1>(exp_record) << ", "
              << std::get<2>(exp_record) << ", "
              << std::get<3>(exp_record) << ", "
              << std::get<4>(exp_record) << ", "
              << std::get<5>(exp_record) << ", "
              << std::get<6>(exp_record) << ", "
              << std::get<7>(exp_record) << std::endl;

    std::cout << "Experiment finished" << std::endl;
}