add_library(DeepLearningPP STATIC dummy.cpp)

target_include_directories(DeepLearningPP PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_include_directories(DeepLearningPP PUBLIC ${PROJECT_SOURCE_DIR}/externals/nameof/include)
