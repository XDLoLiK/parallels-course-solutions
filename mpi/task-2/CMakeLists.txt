cmake_minimum_required(VERSION 3.16)
project(MPI_Task_2 C)

find_package(MPI REQUIRED)

set(TARGET conductivity)

set(SOURCES
    src/main.c
    src/mpi_error.c)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET} PRIVATE include/)
target_link_libraries(${TARGET} MPI::MPI_C)
