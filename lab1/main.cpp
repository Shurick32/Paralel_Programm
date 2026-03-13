#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdlib>

bool readMatrix(const std::string& filename, std::vector<std::vector<double>>& matrix, int& size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть файл " << filename << std::endl;
        return false;
    }

    file >> size;
    if (size <= 0) {
        std::cerr << "Ошибка: Некорректный размер матрицы." << std::endl;
        return false;
    }

    matrix.resize(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (!(file >> matrix[i][j])) {
                std::cerr << "Ошибка: Недостаточно данных в файле " << filename << std::endl;
                return false;
            }
        }
    }
    file.close();
    return true;
}

void multiplyMatrices(const std::vector<std::vector<double>>& A, 
                      const std::vector<std::vector<double>>& B, 
                      std::vector<std::vector<double>>& C, int size) {
    C.resize(size, std::vector<double>(size));
    for (int i = 0; i < size; ++i) {
        for (int k = 0; k < size; ++k) {
            if (A[i][k] == 0) continue;
            for (int j = 0; j < size; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

bool writeResult(const std::string& filename, const std::vector<std::vector<double>>& matrix, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка: Не удалось создать файл результата " << filename << std::endl;
        return false;
    }

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

int main() {
    setlocale(LC_ALL, "");
    std::vector<std::vector<double>> A, B, C;
    int sizeA, sizeB;

    if (!readMatrix("M-a.txt", A, sizeA)) return 1;
    if (!readMatrix("M-b.txt", B, sizeB)) return 1;

    if (sizeA != sizeB) {
        std::cerr << "Ошибка: Матрицы должны быть квадратными и одинакового размера для этой задачи." << std::endl;
        return 1;
    }
    int N = sizeA;

    std::cout << "Задача: Умножение квадратных матриц размером " << N << "x" << N << std::endl;

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    
    multiplyMatrices(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    //Простой замер объёма
    size_t inputVol = 2 * N * N * sizeof(double); //на входе 2 матрицы
    size_t outputVol = N * N * sizeof(double); // результат
    size_t totalVolume = inputVol + outputVol;

    std::cout << "Время выполнения: " << duration.count() << " сек." << std::endl;
    std::cout << "Объем задачи (Вход + Выход): " << totalVolume << " байт (" 
              << (double)totalVolume / 1024.0 << " Кб)" << std::endl;

    if (!writeResult("result.txt", C, N)) return 1;
    std::cout << "Результат сохранен в result.txt" << std::endl;

    std::cout << "\nЗапуск верификации (Python)..." << std::endl;
    
    int verifyStatus = system("python verify.py"); 

    if (verifyStatus == 0) {
        std::cout << "Проверка пройдена: Результаты совпадают." << std::endl;
    } else {
        std::cerr << "Проверка не пройдена или скрипт Python не найден." << std::endl;
        return 1;
    }

    return 0;
}