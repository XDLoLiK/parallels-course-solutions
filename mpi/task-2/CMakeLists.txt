cmake_minimum_required(VERSION 3.16)
project(task2 C)

find_package(MPI REQUIRED)

set(TARGET task2)

set(SOURCES
    src/main.c
    src/mpi_error.c
    src/mpi_print.c)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET} PRIVATE include/)

target_link_libraries(${TARGET} MPI::MPI_C)
