#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>
#include <Eigen/Dense> //verify для C++
#include <mpi.h>
#include <windows.h>

bool readMatrix(const std::string& filename, std::vector<double>& matrix, int& size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return true;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << std::endl;
        return false;
    }
    file >> size;
    if (size <= 0) return false;
    matrix.resize(size * size);
    for (int i = 0; i < size * size; ++i) {
        if (!(file >> matrix[i])) {
            std::cerr << "Ошибка чтения данных в файле: " << filename << std::endl;
            return false;
        }
    }
    file.close();
    return true;
}

bool writeResult(const std::string& filename, const std::vector<double>& matrix, int size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return true;

    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << size << std::endl;
    file << std::fixed << std::setprecision(1);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << matrix[i * size + j] << " ";
        }
        file << std::endl;
    }
    file.close();
    return true;
}

void localMultiply(const double* A_local, const double* B, double* C_local, int rows, int size) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A_local[i * size + k] * B[k * size + j];
            }
            C_local[i * size + j] = sum;
        }
    }
}


bool verify_with_eigen(const std::vector<double>& A_flat, const std::vector<double>& B_flat, const std::vector<double>& C_flat, int n) {
    Eigen::MatrixXd eigenA(n, n);
    Eigen::MatrixXd eigenB(n, n);
    Eigen::MatrixXd eigenC_cpp(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = i * n + j;
            eigenA(i, j) = A_flat[idx];
            eigenB(i, j) = B_flat[idx];
            eigenC_cpp(i, j) = C_flat[idx];
        }
    }
    Eigen::MatrixXd eigenC_ref = eigenA * eigenB;
    double max_diff = (eigenC_ref - eigenC_cpp).cwiseAbs().maxCoeff();

    if (max_diff < 1e-9) {
        std::cout << "Максимальная погрешность: " << max_diff << " (OK)" << std::endl;
        return true;
    }
    else {
        std::cerr << "ОШИБКА! Максимальная погрешность: " << max_diff << std::endl;
        return false;
    }
}

void record_mpi_statistics(std::chrono::duration<double> duration,
    const std::vector<double>& C_flat,
    const std::string& output_file,
    int num_processes) {

    long long total_elements = C_flat.size();
    int n = static_cast<int>(std::sqrt(total_elements));

    if (n * n != total_elements) {
        std::cerr << "[Ошибка] Размер результата не является квадратом целого числа!" << std::endl;
        return;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) return;

    long long operations = 2LL * n * n * n;

    bool write_header = false;
    std::ifstream check(output_file, std::ios::in);
    if (!check.good() || check.peek() == std::ifstream::traits_type::eof()) {
        write_header = true;
    }
    check.close();

    std::ofstream out(output_file, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл для записи статистики: " << output_file << std::endl;
        return;
    }
    if (write_header) {
        out << "Size,Num_processes,Time_Seconds,Operations" << "\n";
    }

    out << n << ","
        << num_processes << ","
        << duration.count() << ","
        << operations << "\n";

    out.flush();
    out.close();

    std::cout << "Статистика записана в " << output_file << std::endl;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(CP_UTF8);

    MPI_Init(&argc, &argv);

    int provided;
    int init_result = MPI_Init(&argc, &argv);

    if (init_result != MPI_SUCCESS) {
        std::cerr << "КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать MPI среду." << std::endl;
        return 1;
    }

    int rank, size_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size_proc);

    int N = 0;
    std::vector<double> A_full, B_full, C_full;
    std::vector<double> A_local, C_local;

    if (rank == 0) {
        if (!readMatrix("m_a.txt", A_full, N)) {
            std::cerr << "Ошибка чтения m_a.txt" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        std::ifstream file_b("m_b.txt");
        int tmp_size;
        file_b >> tmp_size;
        if (tmp_size != N) {
            std::cerr << "Размеры матриц не совпадают!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        B_full.resize(N * N);
        for (int i = 0; i < N * N; ++i) file_b >> B_full[i];
        file_b.close();

        std::cout << "Матрицы загружены. Размер: " << N << "x" << N << std::endl;
        std::cout << "Запуск на " << size_proc << " процессах..." << std::endl;
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N <= 0) { MPI_Finalize(); return 1; }

    B_full.resize(N * N);
    
    MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int base_rows = N / size_proc;
    int remainder = N % size_proc;

    int my_rows = base_rows + (rank < remainder ? 1 : 0);

    int offset = 0;
    for (int i = 0; i < rank; ++i) {
        offset += base_rows + (i < remainder ? 1 : 0);
    }

    A_local.resize(my_rows * N);
    C_local.resize(my_rows * N);

    std::vector<int> sendcounts(size_proc);
    std::vector<int> displs(size_proc);

    if (rank == 0) {
        int current_disp = 0;
        for (int i = 0; i < size_proc; ++i) {
            int r = base_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = r * N;
            displs[i] = current_disp;
            current_disp += sendcounts[i];
        }
    }

    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        A_local.data(), my_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    localMultiply(A_local.data(), B_full.data(), C_local.data(), my_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double exec_time = end_time - start_time;
    std::chrono::duration<double> duration(exec_time);

    C_full.resize(N * N);
    MPI_Gatherv(C_local.data(), my_rows * N, MPI_DOUBLE,
        C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Вычисления завершены." << std::endl;
        std::cout << "Время выполнения (параллельное): " << exec_time << " сек." << std::endl;
        
        size_t volume = (2LL * N * N + 1LL * N * N) * sizeof(double);
        std::cout << "Объем данных: " << volume << " байт" << std::endl;

        writeResult("result_mpi.txt", C_full, N);
        std::cout << "Результат сохранен в result_mpi.txt" << std::endl;
        record_mpi_statistics(duration, C_full, "mpi_benchmark.csv", size_proc);

        std::cout << verify_with_eigen(A_full, B_full, C_full, N) << std::endl;
    }

    MPI_Finalize();
    return 0;
}