#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "mpi_defs.h"
#include "mpi_error.h"
#include "utils.h"

#define L 1.0
#define K 1.0
#define DT 0.0002
#define T 0.1

static int world_size;
static int world_rank;

static void
mpi_init(int argc, const char* argv[])
{
    MPI_CHECK_ERR(MPI_Init(&argc, &argv));

    MPI_CHECK_ERR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
}

static void
mpi_exit()
{
    MPI_Finalize();
}

static double
exact_u(double x, double t)
{
    const double threshold = 1e-8;
    double sum = 0, term;
    int n, m;

    for (m = 0; /* */; m++) {
        n = 2 * m + 1;
        // u0 == k == l == 1
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
exchange(double* u, int size)
{
#if defined(EXCHANGE_SLOW)
    exchange_slow(u, size);
#else
    exchange_fast(u, size);
#endif /* defined(EXCHANGE_SLOW) */
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
print_result(double* full, int global_n)
{
    int n, i;
    double h, x, u;

    for (n = 0; n <= 10; n++) {
        i = n * (global_n - 1) / 10;
        h = L / (global_n - 1);
        x = i * h;

        u = exact_u(x, T);
        printf("x=%.2f numeric=%f exact=%f\n",
            x, full[i], u);
    }
}

int
main(int argc, const char* argv[])
{
    double t0, t1, alpha, h;
    int global_n, local_n, steps;
    double* u = NULL;
    double* u_intermediate = NULL;
    double* full = NULL;

    mpi_init(argc, argv);

    if (argc < 2) {
        // H = 0.02
        global_n = (int)(L / 0.02) + 1;
    } else {
        global_n = atoi(argv[1]);
    }

ROOT_DO(
#if defined(EXCHANGE_SLOW)
    printf("exchange_slow is enabled\n");
#else
    printf("exchange_fast is enabled\n");
#endif /* defined(EXCHANGE_SLOW) */
) /* ROOT_DO */

    local_n = decompose(global_n);

    u = calloc(local_n + 2, sizeof(double));
    u_intermediate = calloc(local_n + 2, sizeof(double));

    init(u, local_n);

    t0 = MPI_Wtime();

    h = L / (global_n - 1);
    steps = (int)(T / DT);
    alpha = K * DT / (h * h);

    for (int t = 0; t < steps; t++) {
        exchange(u, local_n);
        step(u, u_intermediate, local_n, alpha);
    }

    t1 = MPI_Wtime();

ROOT_DO(
    printf("Time: %fs\n", t1 - t0);
) /* ROOT_DO */

ROOT_DO(
    full = calloc(global_n, sizeof(double));
) /* ROOT_DO */

    gather(full, u, local_n);

ROOT_DO(
    print_result(full, global_n);
    free(full);
) /* ROOT_DO */

    free(u);
    free(u_intermediate);

    mpi_exit();

    return EXIT_SUCCESS;
}
