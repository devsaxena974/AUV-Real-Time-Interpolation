#include "../include/Point.h"
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel for bilinear interpolation
 * 
 * @param grid Input elevation grid (flattened 2D array)
 * @param points Input query points
 * @param results Output interpolated values
 * @param num_points Number of points to interpolate
 * @param min_lon Minimum longitude value
 * @param max_lon Maximum longitude value
 * @param min_lat Minimum latitude value
 * @param max_lat Maximum latitude value
 * @param num_lon Number of grid points in longitude direction
 * @param num_lat Number of grid points in latitude direction
 * @param lon_step Grid cell size in longitude direction
 * @param lat_step Grid cell size in latitude direction
 */
__global__ void bilinearInterpolationKernel(
    const double* __restrict__ grid,
    const Point* __restrict__ points,
    double* __restrict__ results,
    int num_points,
    double min_lon, double max_lon,
    double min_lat, double max_lat,
    int num_lon, int num_lat,
    double lon_step, double lat_step
) {
    // Calculate global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process a point
    if (tid < num_points) {
        // Get the point to interpolate
        double lon = points[tid].lon;
        double lat = points[tid].lat;
        
        // Check if the point is inside the grid bounds
        if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
            results[tid] = NAN;  // Return NaN for out-of-bounds points
            return;
        }
        
        // Find the indices of the grid cell that contains the point
        double x = (lon - min_lon) / lon_step;  // Normalized x-coordinate
        double y = (lat - min_lat) / lat_step;  // Normalized y-coordinate
        
        int x0 = floor(x);              // Lower x index
        int y0 = floor(y);              // Lower y index
        int x1 = min(x0 + 1, num_lon - 1);  // Upper x index (boundary check)
        int y1 = min(y0 + 1, num_lat - 1);  // Upper y index (boundary check)
        
        // Calculate interpolation weights
        double wx = x - x0;  // Weight for x interpolation
        double wy = y - y0;  // Weight for y interpolation
        
        // Get the four corner elevations
        // Grid is stored in row-major order (y is the outer index)
        double z00 = grid[y0 * num_lon + x0];  // Bottom-left elevation
        double z01 = grid[y0 * num_lon + x1];  // Bottom-right elevation
        double z10 = grid[y1 * num_lon + x0];  // Top-left elevation
        double z11 = grid[y1 * num_lon + x1];  // Top-right elevation
        
        // Perform bilinear interpolation
        // First interpolate along x direction for both y values
        double z0 = (1 - wx) * z00 + wx * z01;  // Bottom edge interpolation
        double z1 = (1 - wx) * z10 + wx * z11;  // Top edge interpolation
        
        // Then interpolate along y direction
        results[tid] = (1 - wy) * z0 + wy * z1;  // Final interpolated value
    }
}