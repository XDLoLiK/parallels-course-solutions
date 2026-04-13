#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "mpi_defs.h"
#include "mpi_print.h"
#include "mpi_error.h"
#include "utils.h"

#define L 1.0
#define K 1.0
#define SZ 11

#define DEFAULT_FILENAME "data.txt"

#define TIME_IT(expr)                               \
({                                                  \
    double t0, t1, t_loc, t_max;                    \
                                                    \
    t0 = MPI_Wtime();                               \
                                                    \
    expr;                                           \
                                                    \
    t1 = MPI_Wtime();                               \
                                                    \
    t_loc = t1 - t0;                                \
    MPI_CHECK_ERR(MPI_Reduce(&t_loc, &t_max, 1, MPI_DOUBLE,         \
                             MPI_MAX, ROOT_RANK, MPI_COMM_WORLD));  \
    t_max;                                          \
})

/*
 * MPI global params
 */
int world_size;
int world_rank;

/*
 * Command line arguments
 */
static int global_n;
static double global_t;
static FILE* out_file = NULL;

static int
mpi_init(int argc, const char* argv[])
{
    int opt;
    const char* filename = DEFAULT_FILENAME;

    MPI_CHECK_ERR(MPI_Init(&argc, (char***)&argv));

    MPI_CHECK_ERR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    /*
     * Defaults:
     * H = 0.0002
     * T = 0.1
     */
    global_n = (int)(L / 0.0002) + 1;
    global_t = 0.1;

    while ((opt = getopt(argc, (char* const*)argv, "n:t:f:")) != -1) {
        switch (opt) {
        case 'n':
            global_n = atoi(optarg);
            break;
        case 't':
            global_t = atof(optarg);
            break;
        case 'f':
            filename = optarg;
            break;
        default:
            mpi_print("Usage: %s [-n points] [-t time] [-f output_file]\n",
                      argv[0]);
            return -1;
        }
    }

ROOT_DO(
    out_file = fopen(filename, "a");
    if (out_file == NULL) {
        return -1;
    }
) /* ROOT_DO */

    return 0;
}

static int
mpi_deinit()
{
ROOT_DO(
    fclose(out_file);
) /* ROOT_DO */

    MPI_Finalize();

    return 0;
}

static double
exact_u(double x, double t)
{
    const double threshold = 1e-8;
    double sum = 0, term;
    int n, m;

    for (m = 0; /* */; m++) {
        n = 2 * m + 1;
        /* u0 == k == l == 1 */
        term = (4 / (n * M_PI)) * sin(n * M_PI * x) *
            exp(-t * n * n * M_PI * M_PI);
        if (fabs(term) < threshold) {
            return sum;
        }
        sum += term;
    }
}

static int
decompose(int n)
{
    int div = n / world_size;
    int rem = n % world_size;

    return div + (world_rank < rem ? 1 : 0);
}

static void
init(double* u, int size)
{
    int i;

    for (i = 1; i <= size; i++) {
        u[i] = 1;
    }
}

static void
exchange_fast(double* u, int size)
{
    if (world_rank == 0) {
        u[0] = 0;
    }

    if (world_rank > 0) {
        MPI_CHECK_ERR(MPI_Sendrecv(&u[1], 1, MPI_DOUBLE, world_rank - 1, 0,
                                   &u[0], 1, MPI_DOUBLE, world_rank - 1, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    if (world_rank < world_size - 1) {
        MPI_CHECK_ERR(MPI_Sendrecv(&u[size], 1, MPI_DOUBLE, world_rank + 1, 0,
                                   &u[size + 1], 1, MPI_DOUBLE, world_rank + 1, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    if (world_rank == world_size - 1){
        u[size + 1] = 0;
    }
}

static void
exchange_slow(double* u, int size)
{
    MPI_Status status;

    if (world_rank > 0) {
        MPI_CHECK_ERR(MPI_Send(&u[1], 1, MPI_DOUBLE,
                               world_rank - 1, 0,
                               MPI_COMM_WORLD));
    }

    if (world_rank < world_size - 1) {
        MPI_CHECK_ERR(MPI_Recv(&u[size + 1], 1, MPI_DOUBLE,
                               world_rank + 1, 0,
                               MPI_COMM_WORLD, &status));
    }

    if (world_rank < world_size - 1) {
        MPI_CHECK_ERR(MPI_Send(&u[size], 1, MPI_DOUBLE,
                               world_rank + 1, 0,
                               MPI_COMM_WORLD));
    }

    if (world_rank > 0) {
        MPI_CHECK_ERR(MPI_Recv(&u[0], 1, MPI_DOUBLE,
                               world_rank - 1, 0,
                               MPI_COMM_WORLD, &status));
    }

    if (world_rank == 0) {
        u[0] = 0.0;
    }

    if (world_rank == world_size - 1) {
        u[size + 1] = 0.0;
    }
}

static void
step(double* u, double* u_intermediate, int size, double alpha)
{
    int i;

    for (i = 1; i <= size; i++) {
        u_intermediate[i] = u[i] + alpha * (u[i + 1] - 2 * u[i] + u[i - 1]);
    }

    if (world_rank == 0) {
        u_intermediate[1] = 0.0;
    }

    if (world_rank == world_size - 1) {
        u_intermediate[size] = 0.0;
    }

    for (i = 1; i <= size; i++) {
        u[i] = u_intermediate[i];
    }
}

static void
gather(double* full, double* u, int size)
{
    int* counts = NULL;
    int* offsets = NULL;
    int i;

ROOT_DO(
    counts = calloc(world_size, sizeof(int));
    offsets = calloc(world_size, sizeof(int));
) /* ROOT_DO */

    MPI_CHECK_ERR(MPI_Gather(&size, 1, MPI_INT,
                             counts, 1, MPI_INT,
                             ROOT_RANK, MPI_COMM_WORLD));

ROOT_DO(
    offsets[0] = 0;
    for (i = 1; i < world_size; i++) {
        offsets[i] = offsets[i - 1] + counts[i - 1];
    }
) /* ROOT_DO */

    MPI_CHECK_ERR(MPI_Gatherv(&u[1], size, MPI_DOUBLE,
                              full, counts, offsets, MPI_DOUBLE,
                              ROOT_RANK, MPI_COMM_WORLD));

ROOT_DO (
    free(counts);
    free(offsets);
) /* ROOT_DO */
}

static void
print_result(double* numeric, int global_n)
{
    int n, i;
    double h, x, u;
    double exact[SZ];

    h = L / (global_n - 1);

    for (i = 0; i < SZ; i++) {
        n = i * (global_n - 1) / 10;
        x = n * h;
        exact[i] = exact_u(x, global_t);
    }

    for (i = 0; i < SZ; i++) {
        n = i * (global_n - 1) / 10;
        x = n * h;
        mpi_print("%d: x=%.2f numeric=%f exact=%f\n",
                  i, x, numeric[n], exact[i]);
    }
}

int
main(int argc, const char* argv[])
{
    double t, alpha, h, dt;
    int local_n, steps;
    double* u = NULL;
    double* u_intermediate = NULL;
    double* full_slow = NULL;
    double* full_fast = NULL;

    if (mpi_init(argc, argv) < 0) {
        goto out;
    }

    local_n = decompose(global_n);

    u = calloc(local_n + 2, sizeof(double));
    u_intermediate = calloc(local_n + 2, sizeof(double));

ROOT_DO(
    full_slow = calloc(global_n, sizeof(double));
    full_fast = calloc(global_n, sizeof(double));
) /* ROOT_DO */

    h = L / (global_n - 1);
    dt = 0.5 * h * h / K;
    steps = (int)(global_t / dt);
    alpha = K * dt / (h * h);

    if (steps < 1) {
        steps = 1;
    }

#ifdef COMPUTE_SLOW
    t = TIME_IT(
        init(u, local_n);

        for (int t = 0; t < steps; t++) {
            exchange_slow(u, local_n);
            step(u, u_intermediate, local_n, alpha);
        }

        gather(full_slow, u, local_n);
    );

    mpi_fprint(out_file, "%.8f\n", t);
#endif /* COMPUTE_SLOW */

    t = TIME_IT(
        init(u, local_n);

        for (int t = 0; t < steps; t++) {
            exchange_fast(u, local_n);
            step(u, u_intermediate, local_n, alpha);
        }

        gather(full_fast, u, local_n);
    );

    mpi_fprint(out_file, "%.8f\n", t);

ROOT_DO(
    print_result(full_fast, global_n);

    free(full_slow);
    free(full_fast);
) /* ROOT_DO */

    free(u);
    free(u_intermediate);

out:
    mpi_deinit();

    return EXIT_SUCCESS;
}
