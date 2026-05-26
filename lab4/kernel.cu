constexpr bool USE_CMD_ARGS = true;

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool readMatrix(const std::string& filename, std::vector<double>& matrix, int& size) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    file >> size;
    matrix.resize(size * size);
    for (int i = 0; i < size * size; ++i) {
        if (!(file >> matrix[i])) return false;
    }
    return true;
}

bool writeResult(const std::string& filename, const std::vector<double>& matrix, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << size << "\n";
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << matrix[i * size + j] << " ";
        }
        file << "\n";
    }
    return true;
}

bool verify_with_eigen(const std::vector<double>& A, const std::vector<double>& B,
    const std::vector<double>& C, int N) {
    Eigen::MatrixXd ea(N, N), eb(N, N), ec_cpp(N, N);
    for (int i = 0; i < N * N; ++i) {
        ea(i) = A[i];
        eb(i) = B[i];
        ec_cpp(i) = C[i];
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            ea(i, j) = A[i * N + j];
            eb(i, j) = B[i * N + j];
            ec_cpp(i, j) = C[i * N + j];
        }

    Eigen::MatrixXd ref = ea * eb;
    double max_diff = (ref - ec_cpp).cwiseAbs().maxCoeff();
    return max_diff < 1e-8;
}

void record_statistics(const std::string& filename, int N, int block_size_x, int block_size_y, double total_time) {
    std::ofstream out(filename, std::ios::app);
    if (out.tellp() == 0) out << "Size,Block_X,Block_Y,Time_Seconds\n";
    out << N << "," << block_size_x << "," << block_size_y << "," << total_time << "\n";
    out.close();
}

int main() {
    setlocale(LC_ALL, "");
    
    std::vector<double> A, B;
    int N;

    if (!readMatrix("m_a20.txt", A, N)) {
        std::cerr << "Ошибка чтения m_a.txt" << std::endl;
        return 1;
    }
    std::ifstream fb("m_b20.txt");
    int tmp; fb >> tmp;
    if (N == 0 || N != tmp) {
        std::cerr << "Error: Матрицы разного размера.\n";
        return 1;
    }
    B.resize(N * N);
    for (auto& x : B) fb >> x;

    std::cout << "Матрица загружена: " << N << "x" << N << std::endl;

    double* d_A, * d_B, * d_C;
    size_t size_bytes = N * N * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_A, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, size_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), size_bytes, cudaMemcpyHostToDevice));

    // Конфигурации блоков для эксперимента
    std::vector<int> block_sizes = {32};

    std::string csv_file = "cuda_benchmark.csv";

    std::vector<double> C_result(N * N);

    for (int bs : block_sizes) {
        dim3 blockDim(bs, bs);
        dim3 gridDim((N + bs - 1) / bs, (N + bs - 1) / bs);

        CUDA_CHECK(cudaDeviceSynchronize());

        auto start_total = std::chrono::high_resolution_clock::now();

        matrixMulKernel <<<gridDim, blockDim >>> (d_A, d_B, d_C, N);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(C_result.data(), d_C, size_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_dur = end_total - start_total;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        matrixMulKernel <<<gridDim, blockDim >>> (d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        double total_time_ms = total_dur.count();

        std::cout << "Block Size: " << bs << "x" << bs
            << " | Total: " << total_time_ms << " ms" << std::endl;

        if (bs == block_sizes[0]) {
            if (verify_with_eigen(A, B, C_result, N)) {
                std::cout << "[OK] Верификация пройдена." << std::endl;
                writeResult("result_cuda.txt", C_result, N);
            }
            else {
                std::cerr << "[FAIL] Верификация не пройдена!" << std::endl;
            }
        }

        record_statistics(csv_file, N, bs, bs, total_time_ms);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaDeviceReset());

    std::cout << "Тесты завершены. Результаты в " << csv_file << std::endl;
    return 0;
}