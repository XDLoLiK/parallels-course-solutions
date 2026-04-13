import sys
import subprocess
import matplotlib.pyplot as plt

MAX_PROCESSES = 12
RUNS_PER_CONFIG = 1

GRID_SIZES = [2000, 10000, 50000]

DATA_FILE = "data.txt"

IMG_FILE = "img/mpi_speedup.png"


def run_benchmark(binary, grid_size, processes):
    run_times = []

    for run in range(RUNS_PER_CONFIG):
        print(f"Running: grid_size={grid_size}, processes={processes}, run={run + 1}")

        # Clear previous timings
        open(DATA_FILE, "w").close()

        # Run MPI program
        subprocess.run(
            [
                "mpirun",
                "-np",
                str(processes),
                binary,
                "-n",
                str(grid_size),
                "-t",
                "0.0001",
                "-f",
                DATA_FILE,
            ],
            check=True,
        )

        with open(DATA_FILE, "r") as f:
            run_times.append(float(f.readline()))

    # Return average execution time
    return sum(run_times) / len(run_times)


def main(argc, argv):
    if argc < 2:
        print(f"Usage: python3 {argv[0]} <binary_path>")
        return 1

    binary = argv[1]

    execution_times = {}
    speedups = {}

    # Run benchmarks
    for grid_size in GRID_SIZES:
        execution_times[grid_size] = {}

        for processes in range(1, MAX_PROCESSES + 1):
            avg_time = run_benchmark(
                binary,
                grid_size,
                processes,
            )

            execution_times[grid_size][processes] = avg_time

    # Compute speedup
    for grid_size in GRID_SIZES:
        speedups[grid_size] = {}
        t1 = execution_times[grid_size][1]

        for processes in range(1, MAX_PROCESSES + 1):
            tp = execution_times[grid_size][processes]
            speedups[grid_size][processes] = t1 / tp

    # Print results table
    for grid_size in GRID_SIZES:
        print(f"\n===== Grid size = {grid_size} =====")
        print("Processes\tTime (s)\tSpeedup")

        for processes in range(1, MAX_PROCESSES + 1):
            time_value = execution_times[grid_size][processes]
            speedup_value = speedups[grid_size][processes]

            print(f"{processes}\t\t{time_value:.6f}\t{speedup_value:.3f}")

    # Plot speedup graphs
    plt.figure(figsize=(10, 6))

    for grid_size in GRID_SIZES:
        x = list(speedups[grid_size].keys())
        y = list(speedups[grid_size].values())

        plt.plot(
            x,
            y,
            marker="o",
            label=f"N = {grid_size}",
        )

    plt.xlabel("Number of processes p")
    plt.ylabel("Speedup S(p)")
    plt.title("MPI Speedup vs Number of Processes")

    plt.xticks(range(1, MAX_PROCESSES + 1))

    plt.grid(True)

    plt.legend()

    plt.savefig(IMG_FILE)

    print(f"\nGraph saved to {IMG_FILE}")

    plt.show()

    return 0


if __name__ == "__main__":
    argc = len(sys.argv)
    argv = sys.argv

    sys.exit(main(argc, argv))
