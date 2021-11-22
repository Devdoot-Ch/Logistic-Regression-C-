cmake_minimum_required(VERSION 3.14)

project("LogisticRegression" CXX)

add_executable(${PROJECT_NAME} main.cpp)

find_package(NumCpp 2.6.2 REQUIRED)
target_link_libraries(${PROJECT_NAME}
    NumCpp::NumCpp
)

