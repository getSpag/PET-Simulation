cmake_minimum_required(VERSION 3.16)
project(PET VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Typical official OpenCV Windows pack (override with -DOpenCV_DIR or env OpenCV_DIR).
if(WIN32 AND EXISTS "C:/opencv/build/x64/vc16/lib/OpenCVConfig.cmake")
    if(NOT OpenCV_DIR OR OpenCV_DIR STREQUAL "OpenCV_DIR-NOTFOUND")
        set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib" CACHE PATH "Directory containing OpenCVConfig.cmake" FORCE)
    endif()
endif()

find_package(OpenCV REQUIRED)

add_executable(PET main.cpp)
add_executable(PET_new main_new.cpp)

foreach(tgt PET PET_new)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${tgt} PRIVATE ${OpenCV_LIBS})
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /permissive-)
    endif()
    if(WIN32)
        set(_opencv_bin "")
        if(DEFINED OpenCV_BIN_DIR AND EXISTS "${OpenCV_BIN_DIR}")
            set(_opencv_bin "${OpenCV_BIN_DIR}")
        elseif(OpenCV_DIR)
            get_filename_component(_opencv_bin "${OpenCV_DIR}/../bin" ABSOLUTE)
        endif()
        if(_opencv_bin AND EXISTS "${_opencv_bin}")
            file(GLOB _opencv_dlls "${_opencv_bin}/*.dll")
            if(_opencv_dlls)
                add_custom_command(TARGET ${tgt} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_opencv_dlls}
                        $<TARGET_FILE_DIR:${tgt}>
                    COMMAND_EXPAND_LISTS
                )
            endif()
        endif()
        unset(_opencv_bin)
        unset(_opencv_dlls)
    endif()
endforeach()
