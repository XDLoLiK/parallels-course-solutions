#include "integral.h"

#include <mpi.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int world_size;
static int world_rank;

static inline double integrand(double x) {
    return 4 / (1 + pow(x, 2));
}

static inline double compute_integral_step(double x, double delta) {
    double mid = x + delta / 2;
    return integrand(mid) * delta;
}

static inline double compute_integral(int start, int end, double delta) {
    double sum = 0;

    for (int curr = start; curr < end; curr++) {
        double x = INTEGRATION_START + curr * delta;
        sum += compute_integral_step(x, delta);
    }

    return sum;
}

static inline void do_mpi_init(int argc, const char** argv) {
    MPI_Init(&argc, &argv);

    // Get total processes number and the rank of the process we are in
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

int main(int argc, const char** argv) {
    do_mpi_init(argc, argv);

    // The info about segments number is to be read from the user
    int steps_nr = 0;

    // We only read 'steps_nr' if we are the root process
    // and then broadcast the value we read to other processes
    ROOT_DO(
        scanf("%d", &steps_nr);
    );

    MPI_Bcast(&steps_nr, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

    // Do the work and compute the time
    double multiple_time = 0;

    ROOT_DO(
        multiple_time = MPI_Wtime();
    );

    int start = 0;
    int end = 0;

    // Create a partition of the [0, 1] segment
    ROOT_DO(
        int whole = steps_nr / world_size;
        int remainder = steps_nr % world_size;
        int split_start = INTEGRATION_START;
        int split_end = split_start + whole + (remainder-- > 0);

        start = split_start;
        end = split_end;

        for (int rank = 1; rank < world_size; rank++) {
            split_start = split_end;
            split_end = split_start + whole + (remainder-- > 0);

            // Ignore the request return value as all the processes will
            // sync at blocking MPI_Gather call
            MPI_Request req;
            MPI_Send(&split_start, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
            MPI_Send(&split_end,   1, MPI_INT, rank, 0, MPI_COMM_WORLD);
        }
    );

    // Ignore the return status value for now
    NOT_ROOT_DO(
        MPI_Status status;
        MPI_Recv(&start, 1, MPI_INT, ROOT_RANK, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&end,   1, MPI_INT, ROOT_RANK, 0, MPI_COMM_WORLD, &status);
    );

    double delta = (INTEGRATION_END - INTEGRATION_START) / steps_nr;
    double split_res = compute_integral(start, end, delta);

    // Buffers to gather data and time from all the processes
    double *gathered_data = NULL;

    ROOT_DO(
        gathered_data = (double *)calloc(world_size, sizeof(double));
    );

    MPI_Gather(
        &split_res,
        1,
        MPI_DOUBLE,
        gathered_data,
        1,
        MPI_DOUBLE,
        ROOT_RANK,
        MPI_COMM_WORLD
    );

    ROOT_DO(
        multiple_time = MPI_Wtime() - multiple_time;
    );

    // Gather and accumulate all the results in the root process and
    // compute a single process version of the integral
    ROOT_DO(
        double single_time = MPI_Wtime();
        double single_proc = compute_integral(0, steps_nr, delta);
        single_time = MPI_Wtime() - single_time;
        double multiple_proc = 0;

        for (int proc_nr = 0; proc_nr < world_size; proc_nr++) {
            multiple_proc += gathered_data[proc_nr];
            printf("Process %d: %lf\n", proc_nr, gathered_data[proc_nr]);
        }

        // Compare and output the results
        printf("Total multiple processes: %lf in %lfs\n", multiple_proc, multiple_time);
        printf("Total single process: %lf in %lfs\n", single_proc, single_time);

        // Data needed to plot the graph
        FILE *graph_file = fopen("graph_info.txt", "a");

        if (graph_file != NULL) {
            double acceleration = single_time / multiple_time;
            fprintf(graph_file, "%d %lf\n", world_size, acceleration);
            fclose(graph_file);
        }
    );

    free(gathered_data);

    MPI_Finalize();
    return 0;
}
