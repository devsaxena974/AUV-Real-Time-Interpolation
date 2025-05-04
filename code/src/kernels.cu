#include "../include/Point.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// This helper returns the average of non-NaN values from a, b, c, d.
__device__ inline double fallbackAverage(double a, double b, double c, double d) {
    double sum = 0.0;
    int count = 0;
    if (!isnan(a)) { sum += a; count++; }
    if (!isnan(b)) { sum += b; count++; }
    if (!isnan(c)) { sum += c; count++; }
    if (!isnan(d)) { sum += d; count++; }
    return (count > 0) ? sum / count : NAN;
}

__device__ inline double fallbackBilinearFrom4(double p1, double p2, double p3, double p4) {
    return fallbackAverage(p1, p2, p3, p4);
}

//------------------------------------------------------------------------------
// Device function: Search for up to maxCandidates non-NAN neighbors by
// scanning only each “ring” (cells exactly at Chebyshev distance r) until
// we have at least four.  This is O(r) per ring instead of O(r^2).
//------------------------------------------------------------------------------
__device__ int findCandidateNeighbors(
    const double* __restrict__ grid,
    int num_lon, int num_lat,
    double x, double y,               // normalized grid coordinates
    int center_i, int center_j,       // rounded to nearest grid indices
    int maxRadius,
    int maxCandidates,
    int* cand_i, int* cand_j,
    double* cand_val, double* cand_dist
) {
    int count = 0;

    // If the center cell itself is valid, include it first.
    {
        int i = center_i, j = center_j;
        double v = __ldg(&grid[j * num_lon + i]);
        if (!isnan(v) && count < maxCandidates) {
            cand_i[count] = i;
            cand_j[count] = j;
            cand_val[count] = v;
            double di = (i + 0.5) - x, dj = (j + 0.5) - y;
            cand_dist[count] = sqrt(di*di + dj*dj);
            count++;
        }
    }

    // Expand ring by ring
    for (int r = 1; r <= maxRadius && count < maxCandidates; r++) {
        // Top and bottom edges of the ring
        int jTop = center_j - r;
        int jBottom = center_j + r;
        for (int dx = -r; dx <= r && count < maxCandidates; dx++) {
            int i = center_i + dx;
            // Top edge
            if (jTop >= 0 && jTop < num_lat && i >= 0 && i < num_lon) {
                double v = __ldg(&grid[jTop * num_lon + i]);
                if (!isnan(v)) {
                    cand_i[count] = i;
                    cand_j[count] = jTop;
                    cand_val[count] = v;
                    double di = (i + 0.5) - x, dj = (jTop + 0.5) - y;
                    cand_dist[count] = sqrt(di*di + dj*dj);
                    count++;
                    if (count >= maxCandidates) break;
                }
            }
            // Bottom edge (skip corners if r==0, but r>0 here)
            if (jBottom >= 0 && jBottom < num_lat && i >= 0 && i < num_lon) {
                double v = __ldg(&grid[jBottom * num_lon + i]);
                if (!isnan(v)) {
                    cand_i[count] = i;
                    cand_j[count] = jBottom;
                    cand_val[count] = v;
                    double di = (i + 0.5) - x, dj = (jBottom + 0.5) - y;
                    cand_dist[count] = sqrt(di*di + dj*dj);
                    count++;
                    if (count >= maxCandidates) break;
                }
            }
        }
        if (count >= 4) break;

        // Left and right edges (excluding the corners, which we've already done)
        int iLeft = center_i - r;
        int iRight = center_i + r;
        for (int dy = -r + 1; dy <= r - 1 && count < maxCandidates; dy++) {
            int j = center_j + dy;
            // Left edge
            if (iLeft >= 0 && iLeft < num_lon && j >= 0 && j < num_lat) {
                double v = __ldg(&grid[j * num_lon + iLeft]);
                if (!isnan(v)) {
                    cand_i[count] = iLeft;
                    cand_j[count] = j;
                    cand_val[count] = v;
                    double di = (iLeft + 0.5) - x, dj = (j + 0.5) - y;
                    cand_dist[count] = sqrt(di*di + dj*dj);
                    count++;
                    if (count >= maxCandidates) break;
                }
            }
            // Right edge
            if (iRight >= 0 && iRight < num_lon && j >= 0 && j < num_lat) {
                double v = __ldg(&grid[j * num_lon + iRight]);
                if (!isnan(v)) {
                    cand_i[count] = iRight;
                    cand_j[count] = j;
                    cand_val[count] = v;
                    double di = (iRight + 0.5) - x, dj = (j + 0.5) - y;
                    cand_dist[count] = sqrt(di*di + dj*dj);
                    count++;
                    if (count >= maxCandidates) break;
                }
            }
        }
        if (count >= 4) break;
    }

    return count;
}

//------------------------------------------------------------------------------
// Device function: Simple selection sort to pick the four candidates with smallest distance
// This reorders the candidate arrays in-place so that the first 4 are the closest.
//------------------------------------------------------------------------------
__device__ void selectFourNearest(int* cand_i, int* cand_j, double* cand_val, double* cand_dist, int candCount) {
    const int numToSelect = 4;
    for (int m = 0; m < numToSelect; m++) {
        int minIndex = m;
        for (int k = m; k < candCount; k++) {
            if (cand_dist[k] < cand_dist[minIndex])
                minIndex = k;
        }
        // Swap candidate m with candidate minIndex.
        double tempDist = cand_dist[m];
        cand_dist[m] = cand_dist[minIndex];
        cand_dist[minIndex] = tempDist;
        
        int tempI = cand_i[m];
        cand_i[m] = cand_i[minIndex];
        cand_i[minIndex] = tempI;
        
        int tempJ = cand_j[m];
        cand_j[m] = cand_j[minIndex];
        cand_j[minIndex] = tempJ;
        
        double tempVal = cand_val[m];
        cand_val[m] = cand_val[minIndex];
        cand_val[minIndex] = tempVal;
    }
}

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
        double z00 = __ldg(&grid[y0 * num_lon + x0]);  // Bottom-left elevation
        double z01 = __ldg(&grid[y0 * num_lon + x1]);  // Bottom-right elevation
        double z10 = __ldg(&grid[y1 * num_lon + x0]);  // Top-left elevation
        double z11 = __ldg(&grid[y1 * num_lon + x1]);  // Top-right elevation
        
        // // Perform bilinear interpolation
        // // First interpolate along x direction for both y values
        // double z0 = (1 - wx) * z00 + wx * z01;  // Bottom edge interpolation
        // double z1 = (1 - wx) * z10 + wx * z11;  // Top edge interpolation
        
        // // Then interpolate along y direction
        // results[tid] = (1 - wy) * z0 + wy * z1;  // Final interpolated value
        // Check for any NaN values
        if (isnan(z00) || isnan(z01) || isnan(z10) || isnan(z11)) {
            results[tid] = fallbackAverage(z00, z01, z10, z11);
        } else {
            double z0 = (1.0 - wx) * z00 + wx * z01;
            double z1 = (1.0 - wx) * z10 + wx * z11;
            results[tid] = (1.0 - wy) * z0 + wy * z1;
        }
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
    double t2 = t * t;
    double t3 = t2 * t;
    
    return 0.5 * (2.0 * p1 +
                  (-p0 + p2) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
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
    const Point*   __restrict__ points,
    double*        __restrict__ results,
    int num_points,
    double min_lon, double max_lon,
    double min_lat, double max_lat,
    int num_lon, int num_lat,
    double lon_step, double lat_step
) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= num_points) return;

    // 1) load query
    double lon = points[tid].lon;
    double lat = points[tid].lat;
    // 2) bounds check
    if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
        results[tid] = NAN;
        return;
    }

    // 3) map to floating‐point grid coords
    double xf = (lon - min_lon)/lon_step;
    double yf = (lat - min_lat)/lat_step;
    int xi = (int)floor(xf);
    int yi = (int)floor(yf);
    double tx = xf - xi;
    double ty = yf - yi;

    // 4) test 4×4 block for NaNs
    bool anyNaN = false;
    for (int m = -1; m <= 2 && !anyNaN; ++m) {
        int jj = yi + m;
        if (jj < 0)            jj = 0;
        else if (jj >= num_lat) jj = num_lat - 1;
        for (int n = -1; n <= 2; ++n) {
            int ii = xi + n;
            if (ii < 0)             ii = 0;
            else if (ii >= num_lon) ii = num_lon - 1;
            if (isnan(__ldg(&grid[jj*num_lon + ii]))) {
                anyNaN = true;
                break;
            }
        }
    }

    // 5a) no NaNs → ordinary bicubic
    if (!anyNaN) {
        double col[4];
        for (int m = -1; m <= 2; ++m) {
            int jj = yi + m;
            if (jj < 0)            jj = 0;
            else if (jj >= num_lat) jj = num_lat - 1;
            double p[4];
            for (int n = -1; n <= 2; ++n) {
                int ii = xi + n;
                if (ii < 0)             ii = 0;
                else if (ii >= num_lon) ii = num_lon - 1;
                p[n+1] = grid[jj*num_lon + ii];
            }
            col[m+1] = cubicInterpolate(p[0],p[1],p[2],p[3], tx);
        }
        results[tid] = cubicInterpolate(col[0],col[1],col[2],col[3], ty);
        return;
    }

    // 5b) fallback: find nearest non‐NaN neighbors
    const int maxRadius     = 10;
    const int maxCandidates = (2*maxRadius+1)*(2*maxRadius+1);
    int   cand_i[maxCandidates], cand_j[maxCandidates];
    double cand_val[maxCandidates], cand_dist[maxCandidates];

    int found = findCandidateNeighbors(
        grid, num_lon, num_lat,
        xf, yf,
        xi, yi,
        maxRadius, maxCandidates,
        cand_i, cand_j, cand_val, cand_dist
    );

    if (found < 4) {
        // not enough valid neighbors → average whatever we got
        double sum = 0.0;
        for (int k = 0; k < found; ++k) sum += cand_val[k];
        results[tid] = (found>0) ? (sum/found) : NAN;
        return;
    }

    // select the four closest by distance
    selectFourNearest(cand_i, cand_j, cand_val, cand_dist, found);

    // average those four
    results[tid] = fallbackAverage(
        cand_val[0], cand_val[1],
        cand_val[2], cand_val[3]
    );
}


// Define an inline device function for the variogram.
__device__ inline double variogram(double h, double sill, double range) {
    // double nugget = 1e-6;  // A small positive number.
    // return nugget + sill * (1.0 - exp(-h / range));
    double nugget = 1.0;  // Increased from 1e-6 to 1.0 for better numerical conditioning.
    return nugget + sill * (1.0 - exp(-h / range));
}


/**
 * @brief CUDA kernel for ordinary kriging interpolation with adaptive neighbor search.
 * 
 * If one or more of the originally closest 4 grid cells are NAN, the kernel expands the 
 * search region until it finds four valid neighbors and then uses them in the kriging system.
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
    if (tid >= num_points) return;
    
    double q_lon = points[tid].lon;
    double q_lat = points[tid].lat;
    
    if (q_lon < min_lon || q_lon > max_lon || q_lat < min_lat || q_lat > max_lat) {
        results[tid] = NAN;
        return;
    }
    
    // Compute normalized grid coordinates (in floating point index space).
    double x = (q_lon - min_lon) / lon_step;
    double y = (q_lat - min_lat) / lat_step;
    
    // Use the rounded value as the center for the neighbor search.
    int center_i = static_cast<int>(round(x));
    int center_j = static_cast<int>(round(y));
    center_i = center_i < 0 ? 0 : (center_i >= num_lon ? num_lon - 1 : center_i);
    center_j = center_j < 0 ? 0 : (center_j >= num_lat ? num_lat - 1 : center_j);
    
    // Allocate arrays for candidate neighbors.
    const int maxCandidates = 121; // for search radius up to 10 cells (i.e., (2*10+1)^2)
    int cand_i[121], cand_j[121];
    double cand_val[121], cand_dist[121];
    int candCount = findCandidateNeighbors(grid, num_lon, num_lat, x, y, center_i, center_j, 10, maxCandidates,
                                             cand_i, cand_j, cand_val, cand_dist);
    
    if (candCount < 4) {
        // Not enough neighbors found; fallback: average of what we have.
        double sum = 0.0;
        int count = 0;
        for (int k = 0; k < candCount; k++) {
            sum += cand_val[k];
            count++;
        }
        results[tid] = (count > 0) ? (sum / count) : NAN;
        return;
    }
    
    // Select the four nearest neighbors.
    selectFourNearest(cand_i, cand_j, cand_val, cand_dist, candCount);
    
    // Use the first four candidates as the neighbors.
    double neigh_vals[4];
    double coords[4][2];
    for (int k = 0; k < 4; k++) {
        neigh_vals[k] = cand_val[k];
        // Use cell center for physical coordinates:
        double i = (double)cand_i[k];
        double j = (double)cand_j[k];
        coords[k][0] = min_lon + (i + 0.5) * lon_step;
        coords[k][1] = min_lat + (j + 0.5) * lat_step;
    }
    
    // (Optional debugging output for thread 0)
    if (tid == 0) {
        printf("Thread %d: Query lon=%f, lat=%f\n", tid, q_lon, q_lat);
        for (int k = 0; k < 4; k++) {
            printf("Neighbor %d: index=(%d, %d), value=%f, dist=%f, coords=(%f, %f)\n", 
                   k, cand_i[k], cand_j[k], neigh_vals[k], cand_dist[k],
                   coords[k][0], coords[k][1]);
        }
    }
    
    // Build the kriging system using these 4 neighbors.
    // The augmented matrix M has dimensions 5x6.
    double M[5][6];
    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 6; j++){
            M[i][j] = 0.0;
        }
    }
    
    double q[2];
    // For the query point we use its actual coordinates.
    // (Here we assume q_lon and q_lat are at cell centers already or directly as provided)
    q[0] = q_lon;
    q[1] = q_lat;
    
    // Variogram parameters.
    double sill = 100.0;
    double range = 10.0;
    
    // Fill the first 4 rows of the system.
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            double dx = coords[i][0] - coords[j][0];
            double dy = coords[i][1] - coords[j][1];
            double d = sqrt(dx*dx + dy*dy);
            M[i][j] = variogram(d, sill, range);
        }
        M[i][4] = 1.0;
    }
    // Constraint row:
    for (int j = 0; j < 4; j++){
        M[4][j] = 1.0;
    }
    M[4][4] = 0.0;
    
    // Fill in the right-hand side for rows 0-3.
    for (int i = 0; i < 4; i++){
        double dx = coords[i][0] - q[0];
        double dy = coords[i][1] - q[1];
        double d = sqrt(dx*dx + dy*dy);
        M[i][5] = variogram(d, sill, range);
    }
    M[4][5] = 1.0;
    
    // Solve the 5x5 system via Gaussian elimination.
    const int N_system = 5;
    double pivot, factor;
    for (int i = 0; i < N_system; i++){
        pivot = M[i][i];
        if (fabs(pivot) < 1e-12) {
            // If the system is singular, fallback to simply averaging the neighbor values.
            results[tid] = fallbackBilinearFrom4(neigh_vals[0], neigh_vals[1], neigh_vals[2], neigh_vals[3]);
            return;
        }
        for (int j = i; j < N_system+1; j++){
            M[i][j] /= pivot;
        }
        for (int k = 0; k < N_system; k++){
            if (k == i) continue;
            factor = M[k][i];
            for (int j = i; j < N_system+1; j++){
                M[k][j] -= factor * M[i][j];
            }
        }
    }
    double prediction = M[0][5] * neigh_vals[0] + M[1][5] * neigh_vals[1] 
                        + M[2][5] * neigh_vals[2] + M[3][5] * neigh_vals[3];
    results[tid] = prediction;
}

