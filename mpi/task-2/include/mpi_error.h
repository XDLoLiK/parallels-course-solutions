#pragma once

#include <string.h>

#include <mpi.h>

#define MPI_CHECK_ERR(expr)                         \
({                                                  \
    int error;                                      \
    if ((error = (expr)) != MPI_SUCCESS) {          \
        mpi_print_error(error);                     \
        MPI_Finalize();                             \
        exit(EXIT_FAILURE);                         \
    }                                               \
})

void mpi_print_error(int mpi_error);
