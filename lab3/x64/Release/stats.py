import csv
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "mpi_benchmark.csv"

def read_data(input_file):
    grouped_data = defaultdict(lambda: {"processes": [], "times": []})
    processes = []
    times = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        size_column = "Size" 
        for row in reader:
            size = row[size_column]
            processes = int(row["Num_processes"])
            time = float(row["Time_Seconds"])
                
            grouped_data[size]["processes"].append(processes)
            grouped_data[size]["times"].append(time)
    return grouped_data


def plot_graph(processes, times, size_label, output_prefix="res"):
    
    safe_size = str(size_label).replace(" ", "_").replace("*", "x")
    output_file = f"{output_prefix}_{safe_size}.jpg"
    
    plt.figure(figsize=(8, 5))

    plt.plot(processes, times, marker="o", linewidth=2, color="#2E86AB")

    plt.xlabel("Количество процессов")
    plt.ylabel("Время выполнения, с")
    plt.title("Зависимость времени умножения матриц от числа процессов")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(sorted(set(processes)))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    data = read_data(INPUT_FILE)
    for size, values in data.items():
        plot_graph(values["processes"], values["times"], size)