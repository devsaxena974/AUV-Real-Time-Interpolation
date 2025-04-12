#include "../include/GridD.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// Forward declaration of the CUDA kernel
__global__ void bilinearInterpolationKernel(
    const double* __restrict__ grid,
    const Point* __restrict__ points,
    double* __restrict__ results,
    int num_points,
    double min_lon, double max_lon,
    double min_lat, double max_lat,
    int num_lon, int num_lat,
    double lon_step, double lat_step
);

// Constructor
GridD::GridD(double min_longitude, double max_longitude, int longitude_points,
    double min_latitude, double max_latitude, int latitude_points,
    const std::vector<std::vector<double>>& elevation_data) 
    : min_lon(min_longitude), max_lon(max_longitude), 
    min_lat(min_latitude), max_lat(max_latitude),
    num_lon(longitude_points), num_lat(latitude_points),
    d_grid(nullptr), initialized(false) 

{

    // Calculate the grid cell size
    lon_step = (max_lon - min_lon) / (num_lon - 1);
    lat_step = (max_lat - min_lat) / (num_lat - 1);

    // Initialize CUDA
    initialize(elevation_data);
}

// Destructor
GridD::~GridD() {
    cleanup();
}

// Initialize CUDA resources
void GridD::initialize(const std::vector<std::vector<double>>& elevation_data) {
    // Convert 2D grid to flattened 1D array
    std::vector<double> flattened_grid(num_lat * num_lon);
    for (int j = 0; j < num_lat; ++j) {
        for (int i = 0; i < num_lon; ++i) {
            flattened_grid[j * num_lon + i] = elevation_data[j][i];
        }
    }
    
    // Allocate device memory for the grid
    checkCudaErrors(cudaMalloc(&d_grid, sizeof(double) * num_lat * num_lon));
    
    // Copy grid data to device
    checkCudaErrors(cudaMemcpy(d_grid, flattened_grid.data(), 
                              sizeof(double) * num_lat * num_lon, 
                              cudaMemcpyHostToDevice));
    
    initialized = true;
}

// Clean up CUDA resources
void GridD::cleanup() {
    if (initialized && d_grid != nullptr) {
        checkCudaErrors(cudaFree(d_grid));
        d_grid = nullptr;
        initialized = false;
    }
}

// Batch interpolation function for multiple points
std::vector<Point> GridD::batchInterpolate(const std::vector<Point>& query_points) {
    if (!initialized || query_points.empty()) {
        return query_points;  // Return input if not initialized or empty
    }
    
    int num_points = query_points.size();
    std::vector<Point> results = query_points;  // Copy input points
    
    // Allocate device memory for input points and results
    Point* d_points;
    double* d_results;
    
    checkCudaErrors(cudaMalloc(&d_points, sizeof(Point) * num_points));
    checkCudaErrors(cudaMalloc(&d_results, sizeof(double) * num_points));
    
    // Copy points to device
    checkCudaErrors(cudaMemcpy(d_points, query_points.data(), 
                              sizeof(Point) * num_points, 
                              cudaMemcpyHostToDevice));
    
    // Define CUDA kernel configuration
    const int blockSize = 256;
    const int gridSize = (num_points + blockSize - 1) / blockSize;
    
    // Launch kernel
    bilinearInterpolationKernel<<<gridSize, blockSize>>>(
        d_grid, d_points, d_results, num_points,
        min_lon, max_lon, min_lat, max_lat, 
        num_lon, num_lat, lon_step, lat_step
    );
    
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
    
    // Wait for kernel to finish
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy results back to host
    double* h_results = new double[num_points];
    checkCudaErrors(cudaMemcpy(h_results, d_results, sizeof(double) * num_points, 
                              cudaMemcpyDeviceToHost));
    
    // Update elevation values in the result points
    for (int i = 0; i < num_points; ++i) {
        results[i].elev = h_results[i];
    }
    
    // Free temporary host memory
    delete[] h_results;
    
    // Free device memory
    checkCudaErrors(cudaFree(d_points));
    checkCudaErrors(cudaFree(d_results));
    
    return results;
}

// Single point interpolation (performed on CPU for simplicity)
double GridD::interpolate(double lon, double lat) {
    // For single points, using the GPU would have too much overhead
    // Use a vector with one point and call batch interpolate
    std::vector<Point> point = {{lon, lat, 0.0}};
    auto result = batchInterpolate(point);
    return result[0].elev;
}