import random

def generate_matrix(filename, size):
    with open(filename, 'w') as f:
        # Записываем размер матрицы первой строкой
        f.write(f"{size}\n")
        for i in range(size):
            row = []
            for j in range(size):
                # Генерируем случайное число от 0.0 до 10.0
                val = random.uniform(0.0, 10.0)
                row.append(f"{val:.1f}")
            f.write(" ".join(row) + "\n")
    print(f"Файл {filename} создан ({size}x{size}).")

# Размерность
N = 100

generate_matrix("m_a1000.txt", N)
generate_matrix("m_b1000.txt", N)