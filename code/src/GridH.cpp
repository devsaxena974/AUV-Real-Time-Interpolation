#include "../include/GridH.h"
#include <cmath>

GridH::GridH(double max_lat,
    double min_lat,
    int n_lat_points,
    double max_lon,
    double min_lon,
    int n_lon_points,
    std::vector<std::vector<double>>& elevation_data
) : max_lat(max_lat), min_lat(min_lat),
    num_lat(n_lat_points),
    max_lon(max_lon), min_lon(min_lon),
    num_lon(n_lon_points),
    elevations(elevation_data) 
{
    // Calculate grid cell size
    lon_step = (max_lon - min_lon) / (num_lon - 1);
    lat_step = (max_lat - min_lat) / (num_lat - 1);
}

double GridH::bilinearInterpolate(double lon, double lat) const {
    // Check if the point is inside the grid bounds
    if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
        return NAN;  // Return NaN for out-of-bounds points
    }

    // Find the indices of the grid cell that contains the point
    double x = (lon - min_lon) / lon_step;  // Normalized x-coordinate
    double y = (lat - min_lat) / lat_step;  // Normalized y-coordinate

    int x0 = static_cast<int>(std::floor(x));  // Lower x index
    int y0 = static_cast<int>(std::floor(y));  // Lower y index
    int x1 = std::min(x0 + 1, num_lon - 1);    // Upper x index (boundary check)
    int y1 = std::min(y0 + 1, num_lat - 1);    // Upper y index (boundary check)

    // Calculate interpolation weights
    double wx = x - x0;  // Weight for x interpolation
    double wy = y - y0;  // Weight for y interpolation

    // Get the four corner elevations
    double z00 = elevations[y0][x0];  // Bottom-left elevation
    double z01 = elevations[y0][x1];  // Bottom-right elevation
    double z10 = elevations[y1][x0];  // Top-left elevation
    double z11 = elevations[y1][x1];  // Top-right elevation

    // Perform bilinear interpolation
    // First interpolate along x direction for both y values
    double z0 = (1 - wx) * z00 + wx * z01;  // Bottom edge interpolation
    double z1 = (1 - wx) * z10 + wx * z11;  // Top edge interpolation
    
    // Then interpolate along y direction
    double z = (1 - wy) * z0 + wy * z1;     // Final interpolated value

    return z;
}

// ------------------------------------------------------------------
// Cubic interpolation helper: Catmull–Rom spline interpolation
// ------------------------------------------------------------------
static inline double catmullRom(double p0, double p1, double p2, double p3, double t) {
    return 0.5 * (2*p1 + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t*t + (-p0 + 3*p1 - 3*p2 + p3)*t*t*t);
}

// ------------------------------------------------------------------
// Cubic spline interpolation (bicubic) implementation.
// It extracts a 4x4 neighborhood, interpolates first in x then in y.
// ------------------------------------------------------------------
double GridH::cubicInterpolate(double lon, double lat) const {
    if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat) {
        return NAN;
    }
    double x = (lon - min_lon) / lon_step;
    double y = (lat - min_lat) / lat_step;
    
    int x_int = static_cast<int>(std::floor(x));
    int y_int = static_cast<int>(std::floor(y));
    double tx = x - x_int;
    double ty = y - y_int;
    
    double colValues[4];
    for (int m = -1; m <= 2; m++) {
        int j = y_int + m;
        j = std::max(0, std::min(j, num_lat - 1));
        double p[4];
        for (int n = -1; n <= 2; n++) {
            int i = x_int + n;
            i = std::max(0, std::min(i, num_lon - 1));
            p[n + 1] = elevations[j][i];
        }
        colValues[m + 1] = catmullRom(p[0], p[1], p[2], p[3], tx);
    }
    return catmullRom(colValues[0], colValues[1], colValues[2], colValues[3], ty);
}

// ----------------------------------------------------------------
// Ordinary kriging interpolation (CPU):
// This implementation uses the 4 neighboring grid points,
// an exponential variogram model, and solves a 5x5 linear system.
// ----------------------------------------------------------------
double GridH::ordinaryKrigingInterpolate(double lon, double lat) const {
    if (lon < min_lon || lon > max_lon || lat < min_lat || lat > max_lat)
        return NAN;
    
    // Map query point to grid-space.
    double x = (lon - min_lon) / lon_step;
    double y = (lat - min_lat) / lat_step;
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = std::min(x0 + 1, num_lon - 1);
    int y1 = std::min(y0 + 1, num_lat - 1);

    // Define the 4 neighbors (same as bilinear).
    double neighbors[4] = {
        elevations[y0][x0],
        elevations[y0][x1],
        elevations[y1][x0],
        elevations[y1][x1]
    };

    // Compute physical coordinates for the neighbor grid points.
    double coords[4][2] = {
        {min_lon + x0 * lon_step, min_lat + y0 * lat_step},
        {min_lon + x1 * lon_step, min_lat + y0 * lat_step},
        {min_lon + x0 * lon_step, min_lat + y1 * lat_step},
        {min_lon + x1 * lon_step, min_lat + y1 * lat_step}
    };

    double q[2] = {lon, lat};

    // Variogram parameters.
    double sill = 100.0;    // Adjust as needed.
    double range = 10.0;    // Adjust as needed.
    auto variogram = [=](double h) -> double {
        return sill * (1.0 - exp(-h / range));
    };

    // Build augmented 5x5 system.
    // We'll build a matrix M[5][6] where the last column is the right-hand side (b).
    double M[5][6] = {0};

    // Fill rows 0-3.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double dx = coords[i][0] - coords[j][0];
            double dy = coords[i][1] - coords[j][1];
            double d = sqrt(dx * dx + dy * dy);
            M[i][j] = variogram(d);
        }
        M[i][4] = 1.0; // constraint
    }
    // Last row: constraint
    for (int j = 0; j < 4; j++) {
        M[4][j] = 1.0;
    }
    M[4][4] = 0.0;

    // Fill right-hand side: for rows 0-3: b[i] = variogram(distance from neighbor i to q)
    for (int i = 0; i < 4; i++) {
        double dx = coords[i][0] - q[0];
        double dy = coords[i][1] - q[1];
        double d = sqrt(dx*dx + dy*dy);
        M[i][5] = variogram(d);
    }
    M[4][5] = 1.0;

    // Solve the 5x5 system M * lambda = b using Gaussian elimination.
    const int N = 5;
    for (int i = 0; i < N; i++) {
        // Pivot:
        double pivot = M[i][i];
        if (fabs(pivot) < 1e-12) {
            // Fallback to bilinear if singular.
            return bilinearInterpolate(lon, lat);
        }
        for (int j = i; j < N+1; j++) {
            M[i][j] /= pivot;
        }
        // Eliminate i-th column in other rows.
        for (int k = 0; k < N; k++) {
            if (k == i) continue;
            double factor = M[k][i];
            for (int j = i; j < N+1; j++) {
                M[k][j] -= factor * M[i][j];
            }
        }
    }
    // The kriging weights are lambda_i = M[i][5] for i=0..3.
    double prediction = 0;
    for (int i = 0; i < 4; i++) {
        prediction += M[i][5] * neighbors[i];
    }
    return prediction;
}

std::vector<Point> GridH::batchBilinearInterpolate(const std::vector<Point>& query_points) const {
    std::vector<Point> results = query_points;  // Copy input points
    
    // Process each point
    for (auto& point : results) {
        point.elev = bilinearInterpolate(point.lon, point.lat);
    }
    
    return results;
}

std::vector<Point> GridH::batchCubicInterpolate(const std::vector<Point>& query_points) const {
    std::vector<Point> results = query_points;  // copy input points
    for (auto& point : results) {
        point.elev = cubicInterpolate(point.lon, point.lat);
    }
    return results;
}


std::vector<Point> GridH::batchOrdinaryKrigingInterpolate(const std::vector<Point>& query_points) const {
    std::vector<Point> results = query_points;
    for (auto& point : results) {
        point.elev = ordinaryKrigingInterpolate(point.lon, point.lat);
    }
    return results;
}