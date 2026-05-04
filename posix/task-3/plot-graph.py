import sys
import subprocess
import matplotlib.pyplot as plt

MAX_THREADS = 12
RUNS_PER_THREAD = 3
DATA_FILE = "data.txt"


def main(argc, argv):
    if argc < 2:
        print(f"Usage: python3 {argv[0]} <binary_path>")
        return 1

    binary = argv[1]

    # clear old timings
    open(DATA_FILE, "w").close()

    times = {}

    # run benchmarks
    for threads in range(1, MAX_THREADS + 1):
        print(f"Running with {threads} thread(s)...")

        for _ in range(RUNS_PER_THREAD):
            subprocess.run([binary, str(threads)], check=True)

    # read timings
    with open(DATA_FILE, "r") as f:
        values = [float(line.strip()) for line in f.readlines()]

    idx = 0

    for threads in range(1, MAX_THREADS + 1):
        run_times = []

        for _ in range(RUNS_PER_THREAD):
            run_times.append(values[idx])
            idx += 1

        times[threads] = run_times

    # average times
    avg_times = {}

    for threads in range(1, MAX_THREADS + 1):
        avg_times[threads] = sum(times[threads]) / len(times[threads])

    # speedup
    t1 = avg_times[1]

    speedup = {}

    for threads in range(1, MAX_THREADS + 1):
        speedup[threads] = t1 / avg_times[threads]

    # print results
    print("\nResults:")
    print("Threads\tAvg Time (s)\tSpeedup")

    for threads in range(1, MAX_THREADS + 1):
        print(f"{threads}\t{avg_times[threads]:.6f}\t{speedup[threads]:.3f}")

    # plot graph
    x = list(speedup.keys())
    y = list(speedup.values())

    plt.figure(figsize=(10, 6))

    plt.plot(x, y, marker="o")

    plt.xlabel("Number of threads (p)")
    plt.ylabel("Speedup S(p)")
    plt.title("Speedup vs Number of Threads")

    plt.xticks(range(1, MAX_THREADS + 1))
    plt.grid(True)

    plt.savefig("speedup.png")

    print("\nGraph saved to speedup.png")

    plt.show()

    return 0


if __name__ == "__main__":
    argc = len(sys.argv)
    argv = sys.argv

    sys.exit(main(argc, argv))
