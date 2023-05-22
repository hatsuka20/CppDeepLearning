include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip)
FetchContent_MakeAvailable(googletest)

add_executable(test_main test.cpp)
target_link_libraries(test_main gtest_main)
target_link_libraries(test_main DeepLearningPP)

add_test(COMMAND test_main)
