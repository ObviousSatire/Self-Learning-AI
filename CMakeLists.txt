cmake_minimum_required(VERSION 3.20)
project(CoreIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)

add_mlir_dialect_library(CoreIRDialect
  dialects/coreir.td
)
