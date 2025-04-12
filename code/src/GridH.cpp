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

double GridH::interpolate(double lon, double lat) const {
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

std::vector<Point> GridH::batchInterpolate(const std::vector<Point>& query_points) const {
    std::vector<Point> results = query_points;  // Copy input points
    
    // Process each point
    for (auto& point : results) {
        point.elev = interpolate(point.lon, point.lat);
    }
    
    return results;
}