#ifndef MPI_PRINT_H
#define MPI_PRINT_H

#include <stdio.h>
#include <stdarg.h>

void mpi_print(const char* fmt, ...);
void mpi_fprint(FILE* file, const char* fmt, ...);

#endif // MPI_PRINT_H
