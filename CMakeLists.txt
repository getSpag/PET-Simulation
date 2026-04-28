cmake_minimum_required(VERSION 3.16)
project(PET VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional OpenCV hints for common local installs.
#
# Windows users can configure with either:
#   cmake -S . -B build -DOpenCV_DIR="C:/opencv/build/x64/vc16/lib"
# or set the OpenCV_DIR environment variable to the folder containing OpenCVConfig.cmake.
if(NOT DEFINED OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "Path to OpenCVConfig.cmake directory")
endif()

# Homebrew OpenCV (Apple Silicon): helps IDEs / fresh configures find the same install.
if(APPLE AND EXISTS "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
    list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/opencv")
endif()

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

add_executable(PET
    main.cpp
)

add_executable(PET_new
    main_new.cpp
)

function(configure_pet_target target_name)
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(${target_name} PRIVATE ${OpenCV_LIBS})

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /permissive-)
    endif()

    # On Windows, OpenCV DLLs must be next to the executable unless they are on PATH.
    # OpenCV's CMake package commonly exposes OpenCV_BIN_DIR for official builds.
    if(WIN32 AND DEFINED OpenCV_BIN_DIR AND EXISTS "${OpenCV_BIN_DIR}")
        file(GLOB OPENCV_RUNTIME_DLLS "${OpenCV_BIN_DIR}/*.dll")
        if(OPENCV_RUNTIME_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${OPENCV_RUNTIME_DLLS}
                    $<TARGET_FILE_DIR:${target_name}>
                COMMAND_EXPAND_LISTS
                COMMENT "Copying OpenCV runtime DLLs for ${target_name}"
            )
        endif()
    endif()
endfunction()

configure_pet_target(PET)
configure_pet_target(PET_new)
