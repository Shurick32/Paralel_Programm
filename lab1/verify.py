import numpy as np
import sys

def load_matrix(filename):
    try:
        with open(filename, 'r') as f:
            size = int(f.readline().strip())
            data = []
            for line in f:
                row = [float(x) for x in line.split()]
                data.append(row)
            return np.array(data), size
    except Exception as e:
        print(f"Ошибка чтения {filename}: {e}")
        sys.exit(1)

def main():
    A, sizeA = load_matrix('M-a.txt')
    B, sizeB = load_matrix('M-b.txt')

    if sizeA != sizeB:
        print("Размеры матриц не совпадают.")
        sys.exit(1)

    C_cpp, sizeC = load_matrix('result.txt')

    C_python = np.dot(A, B)

    if np.allclose(C_cpp, C_python, rtol=1e-5, atol=1e-8):
        print("Результаты C++ и Python совпадают.")
        sys.exit(0)
    else:
        print("Результаты не совпадают!")
        print("Максимальное расхождение:", np.max(np.abs(C_cpp - C_python)))
        sys.exit(1)

if __name__ == "__main__":
    main()