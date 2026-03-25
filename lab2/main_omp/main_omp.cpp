#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>
#include <Eigen/Dense> //verify для C++
#include <omp.h>

bool readMatrix(const std::string& filename, std::vector<std::vector<double>>& matrix, int& size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << std::endl;
        return false;
    }
    file >> size;
    if (size <= 0) return false;
    matrix.resize(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (!(file >> matrix[i][j])) return false;
    file.close();
    return true;
}

void multiplyMatricesOMP(const std::vector<std::vector<double>>& A, 
                         const std::vector<std::vector<double>>& B, 
                         std::vector<std::vector<double>>& C, int size) {
    C.resize(size, std::vector<double>(size, 0.0));
   
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

bool writeResult(const std::string& filename, const std::vector<std::vector<double>>& matrix, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << size << std::endl;
    file << std::fixed << std::setprecision(1);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << matrix[i][j] << " ";
        }
        file << std::endl;
    }
    file.close();
    return true;
}

void recording_statistics(std::chrono::duration<double> duration,
    const std::vector<std::vector<double>>& C,
    std::string output_file,
    int num_threads) {
    int n = C.size();
    long long operations = 2LL * n * n * n;


    bool write_header = false;
    std::ifstream check(output_file, std::ios::binary);
    if (!check.good() || check.peek() == std::ifstream::traits_type::eof()) {
        write_header = true;
    }
    check.close();

    std::ofstream out(output_file, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Запись отчета в: " << output_file << "\n";
        return;
    }


    if (write_header) {
        out << "Size,Num_threads,Time_Seconds,Operations\n";
    }
    out << n << ","
        << num_threads << ","
        << duration.count() << ","
        << operations << "\n";

    out.flush();
    out.close();
}

bool verify_with_eigen(const std::vector<std::vector<double>>&A, const std::vector<std::vector<double>>&B, const std::vector<std::vector<double>>&C) {
		int n = A.size();
		Eigen::MatrixXd eigenA(n, n);
		Eigen::MatrixXd eigenB(n, n);
		Eigen::MatrixXd eigenC_cpp(n, n);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				eigenA(i, j) = A[i][j];
				eigenB(i, j) = B[i][j];
				eigenC_cpp(i, j) = C[i][j];
			}
		}
		Eigen::MatrixXd eigenC = eigenA * eigenB;
		double max_diff = (eigenC - eigenC_cpp).cwiseAbs().maxCoeff();
		return max_diff < 1e-10;
}

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "");


    #ifdef _OPENMP
    std::cout << "OpenMP успешно включен!" << std::endl;
    #else
    std::cout << "ОШИБКА: OpenMP НЕ включен. Программа будет работать в одном потоке." << std::endl;
    return 1;
    #endif

    std::string resultFile = (argc > 1) ? argv[1] : "result_omp.txt";
    std::string fileA = "m_a.txt";
    std::string fileB = "m_b.txt";

    std::ifstream testA(fileA);
    if (!testA.good()) {
        std::cerr << "Ошибка: Файл " << fileA << " не найден. Сгенерируйте матрицы сначала." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> A, B, C;
    int sizeA, sizeB;

    if (!readMatrix(fileA, A, sizeA)) return 1;
    if (!readMatrix(fileB, B, sizeB)) return 1;

    if (sizeA != sizeB) {
        std::cerr << "Ошибка: Размеры матриц не совпадают." << std::endl;
        return 1;
    }
    int N = sizeA;

    int num_threads = 12; // кол-во потоков, нужно изменять

    omp_set_num_threads(num_threads);

    std::cout << "=== Запуск параллельного умножения (OpenMP) ===" << std::endl;
    std::cout << "Размер матрицы: " << N << "x" << N << std::endl;
    std::cout << "Используемые потоки: " << omp_get_max_threads() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    multiplyMatricesOMP(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Время выполнения: " << duration.count() << " сек." << std::endl;

    if (!writeResult(resultFile, C, N)) {
        std::cerr << "Ошибка записи результата." << std::endl;
        return 1;
    }
    
    std::cout << verify_with_eigen(A, B, C) << std::endl;

    //Проверка verify.py
    /*std::stringstream cmd;
    cmd << "python3 verify.py " << fileA << " " << fileB << " " << resultFile;

    int verifyStatus = system(cmd.str().c_str());

    if (verifyStatus == 0) {
        std::cout << "Проверка пройдена: Результаты совпадают." << std::endl;
    }
    else {
        std::cerr << "Проверка не пройдена или скрипт Python не найден." << std::endl;
        return 1;
    }*/

    std::cout << "Результат сохранен в " << resultFile << std::endl;

    recording_statistics(duration, C, "result_statistic.csv", num_threads);

    return 0;
}