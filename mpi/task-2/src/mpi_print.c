#include "mpi_print.h"
#include "mpi_defs.h"

void
mpi_print(const char* fmt, ...)
{
    va_list args;

NOT_ROOT_DO(
    return;
) /* NOT_ROOT_DO */

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

void
mpi_fprint(FILE* file, const char* fmt, ...)
{
    va_list args;

NOT_ROOT_DO(
    return;
) /* NOT_ROOT_DO */

    va_start(args, fmt);
    vfprintf(file, fmt, args);
    va_end(args);
}
