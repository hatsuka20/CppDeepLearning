add_library(CppDeepLearning STATIC dummy.cpp)

file(GLOB_RECURSE INCLUDES "*.h*")
target_include_directories(CppDeepLearning PUBLIC INCLUDES)
