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