import random
import os

def generate_matrix(filename, size):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    
    with open(full_path, 'w') as f:
        f.write(f"{size}\n")
        for i in range(size):
            row = []
            for j in range(size):
                val = random.uniform(0, 10)
                row.append(f"{val:.2f}")
            f.write(" ".join(row) + "\n")
    print(f"Файл {filename} создан ({size}x{size}).")

# Размерность
N = 2000

generate_matrix("m_a20.txt", N)
generate_matrix("m_b20.txt", N)