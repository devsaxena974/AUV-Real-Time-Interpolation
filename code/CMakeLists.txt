cmake_minimum_required(VERSION 3.18)  # Use at least 3.18 for good CUDA support
project(EdgeApp LANGUAGES CXX CUDA)

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")

# Add source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

# # Add include directories
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# # Create the executable
# add_executable(edge_app ${SOURCES})

# # Set CUDA arch (for your MX550, compute_75 is OK)
# set_target_properties(edge_app PROPERTIES CUDA_ARCHITECTURES 75)

# # Fix .cu file handling under Visual Studio
# set_target_properties(edge_app PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RUNTIME_LIBRARY Static
# )
# Define executables:
# 1. edge_app -> Uses main.cpp along with LIB_SOURCES (if needed)
add_executable(edge_app
    main.cpp
    ${SOURCES}
)

# 2. csv_grid_generator -> Built from generate_csv_grids.cpp and SOURCES
add_executable(csv_grid_generator
    generate_csv_grids.cpp
    ${SOURCES}
)

# 3. test_interpolation -> Built from test_interpolation.cpp and SOURCES
add_executable(test_interpolation
    test_interpolation.cpp
    ${SOURCES}
)

# 4. test_gebco -> Built from test_gebco.cpp and SOURCES
add_executable(test_gebco
    test_gebco.cpp
    ${SOURCES}
)

# Set CUDA properties for each executable
foreach(target edge_app csv_grid_generator test_interpolation)
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES 75            # Change if you have another GPU
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static
    )
endforeach()
