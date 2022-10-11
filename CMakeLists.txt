cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wfatal-errors -Wundef")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(ENV{CMAKE_EXPORT_COMPILE_COMMANDS} ON)

option(CLANG_TIDY "Perform Clang-Tidy" OFF)
option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)

if(CLANG_TIDY OR CLANG_TIDY_FIX)
    find_program(
        CLANG_TIDY_EXE
        NAMES "clang-tidy"
        DOC "Path to clang-tidy executable")

    if(CLANG_TIDY_EXE)
        if(CLANG_TIDY_FIX)
            set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "-fix")
        else()
            set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
        endif()
    endif()
endif()

project(DeeplearningPP CXX)

add_subdirectory(lib)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(
    main PUBLIC ${PROJECT_SOURCE_DIR}/externals/nameof/include)
target_link_libraries(main DeepLearningPP)

enable_testing()
add_subdirectory(test)
