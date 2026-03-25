import csv
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "result_statistic.csv"

def read_data(input_file):
    grouped_data = defaultdict(lambda: {"threads": [], "times": []})
    threads = []
    times = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        size_column = "Size" 
        for row in reader:
            size = row[size_column]
            threads = int(row["Num_threads"])
            time = float(row["Time_Seconds"])
                
            grouped_data[size]["threads"].append(threads)
            grouped_data[size]["times"].append(time)
    return grouped_data


def plot_graph(threads, times, size_label, output_prefix="res"):
    
    safe_size = str(size_label).replace(" ", "_").replace("*", "x")
    output_file = f"{output_prefix}_{safe_size}.jpg"
    
    plt.figure(figsize=(8, 5))

    plt.plot(threads, times, marker="o", linewidth=2, color="#2E86AB")

    plt.xlabel("Количество потоков")
    plt.ylabel("Время выполнения, с")
    plt.title("Зависимость времени умножения матриц от числа потоков")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xticks(sorted(set(threads)))
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    data = read_data(INPUT_FILE)
    for size, values in data.items():
        plot_graph(values["threads"], values["times"], size)