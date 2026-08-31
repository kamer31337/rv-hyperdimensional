cmake_minimum_required(VERSION 3.10)
project(rv_hdc_sequence LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIB_SOURCES
    src/rv_hdc_core.c
    src/rv_hdc_fpe_a.c
    src/rv_hdc_fpe_b.c
    src/rv_hdc_fpe_c.c
    src/rv_hdc_sequence.c
)

add_library(rv_hdc STATIC ${LIB_SOURCES})

if(NOT MSVC)
    target_link_libraries(rv_hdc PUBLIC m)
endif()

add_executable(rv_hdc_demo examples/demo.c)
target_link_libraries(rv_hdc_demo PRIVATE rv_hdc)

if(NOT MSVC)
    target_link_libraries(rv_hdc_demo PRIVATE m)
endif()
