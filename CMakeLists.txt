cmake_minimum_required(VERSION 3.16)
project(PET VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew OpenCV (Apple Silicon): helps IDEs / fresh configures find the same install -- MY PATH WAS FUCKED UP, somehow my x86 install was taking precedence over the arm64 install
if(APPLE AND EXISTS "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
    list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv")
endif()

find_package(OpenCV REQUIRED)

add_executable(PET
    main.cpp
)

target_include_directories(PET PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PET PRIVATE ${OpenCV_LIBS})
