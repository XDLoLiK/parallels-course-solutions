from matplotlib import pyplot
from os import system, remove

segments_nr = [10**3, 10**6, 10**8]
process_nr = list(range(1, 9))

executable = "./task1_mpi.out"
graph_file = "graph_info.txt"

for n in segments_nr:
    for p in process_nr:
        tmp_name = "tmp" + str(n) + str(p) + ".txt"
        tmp = open(tmp_name, "w")
        tmp.write(str(n))
        tmp.close()
        system("mpirun -np " + str(p) + " " + executable + " < " + tmp_name)
        remove(tmp_name)

# processes count
x_axis = []
for i in range(len(segments_nr)):
    sublist = []
    for j in range(len(process_nr)):
        sublist.append(0)
    x_axis.append(sublist)

# acceleration
y_axis = []
for i in range(len(segments_nr)):
    sublist = []
    for j in range(len(process_nr)):
        sublist.append(0)
    y_axis.append(sublist)

with open(graph_file, "r") as data:
    for i in range(len(segments_nr)):
        for j in range(len(process_nr)):
            line = list(map(float, data.readline().split()))
            x_axis[i][j] = line[0]
            y_axis[i][j] = line[1]

pyplot.plot(x_axis[0], y_axis[0], color="r", label="N = 10^3")
pyplot.plot(x_axis[1], y_axis[1], color="g", label="N = 10^6")
pyplot.plot(x_axis[2], y_axis[2], color="b", label="N = 10^8")

pyplot.xlabel("Number of processes")
pyplot.ylabel("Acceleration")
pyplot.legend()

pyplot.savefig("acceleration.png")
