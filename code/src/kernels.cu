#include "../include/Point.h"
#include <cuda_runtime.h>
#include <math.h>

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

/**
 * @brief Device function for cubic interpolation using the Catmull-Rom formula.
 * 
 * @param p0 Value at x = -1
 * @param p1 Value at x =  0
 * @param p2 Value at x = +1
 * @param p3 Value at x = +2
 * @param t  Fractional distance between p1 and p2
 * @return double Interpolated value
 */
__device__ double cubicInterpolate(double p0, double p1, double p2, double p3, double t) {
    return 0.5 * (2.0 * p1 +
                  (-p0 + p2) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
}

/**
 * @brief CUDA kernel for cubic (bicubic) interpolation.
 * 
 * Computes the interpolated value using a 4x4 neighborhood and 
 * the Catmull-Rom cubic interpolation in both x and y directions.
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
__global__ void cubicInterpolationKernel(
    const double* __restrict__ grid,
    const Point* __restrict__ points,
    double* __restrict__ results,
    int num_points,
    double min_lon, double max_lon,
    double min_lat, double max_lat,
    int num_lon, int num_lat,
    double lon_step, double lat_step
) {
    // Calculate the global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_points) {
        double lon = points[tid].lon;
        double lat = points[tid].lat;
        
        // Return NaN for out-of-bounds query points
        if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
            results[tid] = NAN;
            return;
        }
        
        // Map (lon, lat) to grid (x, y) coordinates.
        double x = (lon - min_lon) / lon_step;
        double y = (lat - min_lat) / lat_step;
        int x_int = floor(x);
        int y_int = floor(y);
        double tx = x - x_int;
        double ty = y - y_int;

        // For cubic interpolation we need a 4x4 block of neighbors.
        double interpRows[4];  // Stores interpolation result for each of 4 rows.
        
        // Loop over four rows, offset by -1, 0, +1, +2 relative to y_int.
        for (int m = -1; m <= 2; m++) {
            int j = y_int + m;
            // Clamp to the valid row range.
            j = (j < 0) ? 0 : (j >= num_lat ? num_lat - 1 : j);
            
            double p[4];  // This holds four consecutive grid values in the row.
            // Loop over four columns, offset by -1, 0, +1, +2 relative to x_int.
            for (int n = -1; n <= 2; n++) {
                int i = x_int + n;
                // Clamp to the valid column range.
                i = (i < 0) ? 0 : (i >= num_lon ? num_lon - 1 : i);
                p[n + 1] = grid[j * num_lon + i];
            }
            // Interpolate along the x direction for this row.
            interpRows[m + 1] = cubicInterpolate(p[0], p[1], p[2], p[3], tx);
        }
        
        // Now perform the cubic interpolation along the y direction.
        results[tid] = cubicInterpolate(interpRows[0], interpRows[1], interpRows[2], interpRows[3], ty);
    }
}

// Define an inline device function for the variogram.
__device__ inline double variogram(double h, double sill, double range) {
    return sill * (1.0 - exp(-h / range));
}

/**
 * @brief CUDA kernel for ordinary kriging interpolation.
 * Uses 4 neighboring points and an exponential variogram model.
 */
__global__ void krigingInterpolationKernel(
    const double* __restrict__ grid,
    const Point* __restrict__ points,
    double* __restrict__ results,
    int num_points,
    double min_lon, double max_lon,
    double min_lat, double max_lat,
    int num_lon, int num_lat,
    double lon_step, double lat_step
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_points) {
        double lon = points[tid].lon;
        double lat = points[tid].lat;
        if(lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
            results[tid] = NAN;
            return;
        }
        
        double x = (lon - min_lon) / lon_step;
        double y = (lat - min_lat) / lat_step;
        int x0 = floor(x);
        int y0 = floor(y);
        int x1 = min(x0 + 1, num_lon - 1);
        int y1 = min(y0 + 1, num_lat - 1);
        
        // Get 4 neighbor elevations.
        double neighbors[4];
        neighbors[0] = grid[y0 * num_lon + x0];
        neighbors[1] = grid[y0 * num_lon + x1];
        neighbors[2] = grid[y1 * num_lon + x0];
        neighbors[3] = grid[y1 * num_lon + x1];
        
        // Compute neighbor coordinates.
        double coords[4][2];
        coords[0][0] = min_lon + x0 * lon_step;
        coords[0][1] = min_lat + y0 * lat_step;
        coords[1][0] = min_lon + x1 * lon_step;
        coords[1][1] = min_lat + y0 * lat_step;
        coords[2][0] = min_lon + x0 * lon_step;
        coords[2][1] = min_lat + y1 * lat_step;
        coords[3][0] = min_lon + x1 * lon_step;
        coords[3][1] = min_lat + y1 * lat_step;
        
        double q[2] = {lon, lat};
        
        // Variogram parameters.
        double sill = 100.0;
        double range = 10.0;
        
        // Build augmented 5x6 matrix (5 rows, 5 system coefficients, 6th column is right-hand side).
        double M[5][6] = {0};
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 6; j++) {
                M[i][j] = 0.0;
            }
        }
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double dx = coords[i][0] - coords[j][0];
                double dy = coords[i][1] - coords[j][1];
                double d = sqrt(dx*dx + dy*dy);
                M[i][j] = variogram(d, sill, range);
            }
            M[i][4] = 1.0;
        }
        for (int j = 0; j < 4; j++) {
            M[4][j] = 1.0;
        }
        M[4][4] = 0.0;
        
        for (int i = 0; i < 4; i++) {
            double dx = coords[i][0] - q[0];
            double dy = coords[i][1] - q[1];
            double d = sqrt(dx*dx + dy*dy);
            M[i][5] = variogram(d, sill, range);
        }
        M[4][5] = 1.0;
        
        // Solve the 5x5 system via Gaussian elimination.
        const int N = 5;
        for (int i = 0; i < N; i++) {
            double pivot = M[i][i];
            if (fabs(pivot) < 1e-12) {
                results[tid] = NAN;
                return;
            }
            for (int j = i; j < N+1; j++) {
                M[i][j] /= pivot;
            }
            for (int k = 0; k < N; k++) {
                if (k == i) continue;
                double factor = M[k][i];
                for (int j = i; j < N+1; j++) {
                    M[k][j] -= factor * M[i][j];
                }
            }
        }
        double prediction = 0.0;
        for (int i = 0; i < 4; i++) {
            prediction += M[i][5] * neighbors[i];
        }
        results[tid] = prediction;
    }
}