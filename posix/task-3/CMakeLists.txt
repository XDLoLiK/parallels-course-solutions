cmake_minimum_required(VERSION 3.14)

project(task3 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(task3 main.c)

find_package(Threads REQUIRED)

target_link_libraries(task3
    PRIVATE
    Threads::Threads
    m
)