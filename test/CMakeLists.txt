find_package(GTest REQUIRED)

add_executable(test_main test.cpp)
target_link_libraries(test_main GTest::Main)

target_include_directories(test_main PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_main CppDeepLearning)

add_test(COMMAND test_main)
