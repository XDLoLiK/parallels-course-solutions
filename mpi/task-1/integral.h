#ifndef CALC_INT_H
#define CALC_INT_H

const int ROOT_RANK = 0;
const double INTEGRATION_START = 0.0;
const double INTEGRATION_END   = 1.0;

#define ROOT_DO(expr)               \
do {                                \
    if (world_rank == ROOT_RANK) {  \
        expr                        \
    }                               \
} while (0)

#define NOT_ROOT_DO(expr)           \
do {                                \
    if (world_rank != ROOT_RANK) {  \
        expr                        \
    }                               \
} while (0)

#endif // CALC_INT_H
