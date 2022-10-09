add_library(DeepLearningPP STATIC dummy.cpp)

file(GLOB_RECURSE INCLUDES "*.h*")
target_include_directories(DeepLearningPP PUBLIC INCLUDES)
target_include_directories(
  DeepLearningPP PUBLIC ${PROJECT_SOURCE_DIR}/externals/nameof/include)
