#pragma once

#include <mpi.h>

#define ROOT_RANK 0

#define ROOT_DO(expr)                   \
    if (world_rank == ROOT_RANK) {      \
        expr                            \
    }                                   \

#define NOT_ROOT_DO(expr)               \
    if (world_rank != ROOT_RANK) {      \
        expr                            \
    }                                   \
