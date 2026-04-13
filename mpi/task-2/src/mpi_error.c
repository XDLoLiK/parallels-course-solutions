#include <stdio.h>

#include "mpi_error.h"

void
mpi_print_error(int mpi_error)
{
    int len, error;
    char error_string[MPI_MAX_ERROR_STRING];

    memset(error_string, 0, sizeof(error_string));

    error = MPI_Error_string(mpi_error, error_string, &len);
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Error_string failed (%d), error=%d\n", error, mpi_error);
    } else {
        fprintf(stderr, "%s\n", error_string);
    }
}
