#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <random>
#include <cmath>
#include "include/Point.h"
#include "include/GridH.h"
#include "include/GridD.h"

/**
 * @brief Generate a synthetic bathymetry grid for testing
 * 
 * @param num_lon Number of longitude points
 * @param num_lat Number of latitude points
 * @return std::vector<std::vector<double>> 2D grid of elevation values
 */
std::vector<std::vector<double>> generateTestGrid(int num_lon, int num_lat) {
    std::vector<std::vector<double>> elevations(num_lat, std::vector<double>(num_lon));
    
    // Fill with sample data (a simple pattern in this example)
    for (int j = 0; j < num_lat; ++j) {
        for (int i = 0; i < num_lon; ++i) {
            // Create a depth pattern with some variation
            elevations[j][i] = -1000.0 - 10.0 * sin(i * 0.01) - 15.0 * cos(j * 0.01);
        }
    }
    
    return elevations;
}

// ----------------------------------------------------------------
// Helper Function: Read CSV file into a 2D vector of doubles
// ----------------------------------------------------------------
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> grid;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);  // Default to 0.0 on conversion error
            }
        }
        if (!row.empty()) {
            grid.push_back(row);
        }
    }
    file.close();
    return grid;
}

/**
 * @brief Generate random test points within the specified bounds
 * 
 * @param num_points Number of points to generate
 * @param min_lon Minimum longitude
 * @param max_lon Maximum longitude
 * @param min_lat Minimum latitude
 * @param max_lat Maximum latitude
 * @return std::vector<BathyPoint> Generated test points
 */
std::vector<Point> generateTestPoints(int num_points, 
                                          double min_lon, double max_lon,
                                          double min_lat, double max_lat) {
    std::vector<Point> points(num_points);
    
    // Random number generator
    constexpr double epsilon = 1e-6;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> lon_dist(min_lon + epsilon, max_lon - epsilon);
    std::uniform_real_distribution<> lat_dist(min_lat + epsilon, max_lat - epsilon);
    
    // Generate random points
    for (int i = 0; i < num_points; ++i) {
        points[i] = {lon_dist(gen), lat_dist(gen), 0.0};
    }
    
    return points;
}

// ----------------------------------------------------------------
// Helper Function: Generate an expanded grid of query points
// that covers the same geographic extents but with increased resolution.
// The new grid dimensions are defined as:
//    new_num_lon = 2*num_lon - 1
//    new_num_lat = 2*num_lat - 1
//
// This expanded grid contains both original grid nodes (at even indices)
// and interpolated values (at positions where at least one index is odd).
// ----------------------------------------------------------------
std::vector<Point> generateExpandedGridQueryPoints(int num_lon, int num_lat,
        double min_lon, double max_lon,
        double min_lat, double max_lat) {
    int new_num_lon = 2 * num_lon - 1;
    int new_num_lat = 2 * num_lat - 1;
    std::vector<Point> points;
    points.reserve(new_num_lon * new_num_lat);

    for (int j = 0; j < new_num_lat; ++j) {
        // Compute lat based on new resolution.
        double lat = min_lat + j * (max_lat - min_lat) / (new_num_lat - 1);
        for (int i = 0; i < new_num_lon; ++i) {
            // Compute lon based on new resolution.
            double lon = min_lon + i * (max_lon - min_lon) / (new_num_lon - 1);
            points.push_back({lon, lat, 0.0});
        }
    }
    return points;
}

/**
 * @brief Run benchmark comparing CPU and GPU implementations
 */
void runBenchmark() {
    std::cout << "Bilinear Interpolation Benchmark: CPU vs. CUDA" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    
    // Define grid dimensions
    const int num_lon = 1000;  // 1000 points in longitude
    const int num_lat = 800;   // 800 points in latitude
    double min_lon = -180.0;
    double max_lon = -160.0;
    double min_lat = 20.0;
    double max_lat = 30.0;
    
    std::cout << "Grid dimensions: " << num_lon << "x" << num_lat << " points" << std::endl;
    
    // Create a sample bathymetry grid
    std::cout << "Creating bathymetry grid..." << std::endl;
    auto elevations = generateTestGrid(num_lon, num_lat);
    
    // Create both CPU and GPU bathymetry grid instances
    std::cout << "Initializing CPU and GPU implementations..." << std::endl;
    GridH cpuGrid(max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, elevations);
    
    GridD gpuGrid(min_lon, max_lon, num_lon, 
                              min_lat, max_lat, num_lat, elevations);
    
    // Try different batch sizes to demonstrate scaling
    std::vector<int> batch_sizes = {1000, 10000, 100000, 1000000};
    
    for (int batch_size : batch_sizes) {
        std::cout << "\nTesting with " << batch_size << " points:" << std::endl;
        
        // Generate random test points
        auto testPoints = generateTestPoints(batch_size, min_lon, max_lon, min_lat, max_lat);
        
        // CPU Implementation timing
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpuGrid.batchInterpolate(testPoints);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            cpu_end - cpu_start).count();
        
        // GPU Implementation timing
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = gpuGrid.batchInterpolate(testPoints);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            gpu_end - gpu_start).count();
        
        // Print results
        std::cout << "  CPU time: " << cpu_duration << " ms" << std::endl;
        std::cout << "  GPU time: " << gpu_duration << " ms" << std::endl;
        
        if (gpu_duration > 0) { // Avoid division by zero
            double speedup = static_cast<double>(cpu_duration) / gpu_duration;
            std::cout << "  Speedup: " << speedup << "x" << std::endl;
        }
        
        // Validate results (check a few points)
        bool results_match = true;
        int check_count = std::min(10, batch_size);
        
        for (int i = 0; i < check_count; ++i) {
            double diff = std::abs(cpu_results[i].elev - gpu_results[i].elev);
            if (diff > 1e-6) {
                results_match = false;
                std::cout << "  Results mismatch at point " << i << ": " 
                          << "CPU=" << cpu_results[i].elev 
                          << ", GPU=" << gpu_results[i].elev << std::endl;
                break;
            }
        }
        
        if (results_match) {
            std::cout << "  Results validation: PASSED" << std::endl;
        } else {
            std::cout << "  Results validation: FAILED" << std::endl;
        }
    }
    
    std::cout << "\nBenchmark completed successfully." << std::endl;
}

/**
 * @brief Example showing basic usage of the CPU implementation
 */
void cpuExample() {
    std::cout << "\nCPU Implementation Example:" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    // Create a small grid for demonstration
    const int num_lon = 10;
    const int num_lat = 8;
    double min_lon = -180.0;
    double max_lon = -160.0;
    double min_lat = 20.0;
    double max_lat = 30.0;
    
    // Create a sample bathymetry grid
    std::vector<std::vector<double>> elevations(num_lat, std::vector<double>(num_lon));
    
    // Fill with sample data (a simple slope in this example)
    for (int j = 0; j < num_lat; ++j) {
        for (int i = 0; i < num_lon; ++i) {
            // Create a simple depth pattern (deeper as we go east and north)
            elevations[j][i] = -100.0 - i * 10.0 - j * 5.0;
        }
    }
    
    // Create the bathymetry grid
    GridH bathyGrid(max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, elevations);
    
    // Define some points to interpolate
    std::vector<Point> testPoints = {
        {-175.0, 25.0, 0.0},  // Will be interpolated
        {-170.0, 22.5, 0.0},  // Will be interpolated
        {-165.0, 27.5, 0.0}   // Will be interpolated
    };
    
    // Perform batch interpolation
    auto results = bathyGrid.batchInterpolate(testPoints);
    
    // Print results
    std::cout << "Interpolation Results:" << std::endl;
    for (const auto& point : results) {
        std::cout << "At lon=" << point.lon << ", lat=" << point.lat 
                  << ", depth=" << point.elev << " meters" << std::endl;
    }
    
    // Single point interpolation example
    double test_lon = -172.5;
    double test_lat = 26.25;
    double depth = bathyGrid.interpolate(test_lon, test_lat);
    
    std::cout << "Single point: At lon=" << test_lon << ", lat=" << test_lat 
              << ", depth=" << depth << " meters" << std::endl;
}

// Writes the interpolated grid values to a CSV file.
// Assumes points are stored row-major (num_lon values per row, num_lat rows).
void writeCSV(const std::string& filename, const std::vector<Point>& points,
        int gridCols, int gridRows) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }
    for (int j = 0; j < gridRows; ++j) {
        for (int i = 0; i < gridCols; ++i) {
            int idx = j * gridCols + i;
            file << points[idx].elev;
            if (i < gridCols - 1)
                file << ",";
        }
        file << "\n";
    }
    file.close();
}

/**
 * @brief Program entry point
 */
int main() {
    try {
        // Run the CPU example
        cpuExample();
        
        // Run the benchmark
        runBenchmark();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
// ----------------------------------------------------------------
// Main Function
// ----------------------------------------------------------------
// int main() {
//     try {
//         // Print the current working directory for debugging
//         //std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

//         // Read raw grid data from CSV (e.g., "grid_data.csv")
//         std::string csvFilename = "C:/College/EdgeComputing/grid_data.csv";
//         auto rawGrid = readCSV(csvFilename);
        
//         // Determine grid dimensions from CSV
//         int num_lat = rawGrid.size();
//         if (num_lat == 0) {
//             throw std::runtime_error("The grid data is empty!");
//         }
//         int num_lon = rawGrid[0].size();
//         std::cout << "CSV grid dimensions: " << num_lon << " columns x " << num_lat << " rows" << std::endl;
        
//         // Define geographic extents.
//         // These must match the assumptions of how the grid was generated.
//         double min_lon = -180.0, max_lon = -160.0;
//         double min_lat = 20.0,  max_lat = 30.0;
        
//         // Create CPU and GPU grid objects.
//         // Note:
//         //  - GridH constructor: (max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, elevation_data)
//         //  - GridD constructor: (min_lon, max_lon, num_lon, min_lat, max_lat, num_lat, elevation_data)
//         GridH cpuGrid(max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, rawGrid);
//         GridD gpuGrid(min_lon, max_lon, num_lon, min_lat, max_lat, num_lat, rawGrid);
        
//         // Generate the expanded query grid points.
//         auto queryPoints = generateExpandedGridQueryPoints(num_lon, num_lat, min_lon, max_lon, min_lat, max_lat);
//         // New grid dimensions:
//         int new_num_lon = 2 * num_lon - 1;
//         int new_num_lat = 2 * num_lat - 1;
        
//         // Benchmark CPU interpolation
//         auto cpuStart = std::chrono::high_resolution_clock::now();
//         auto cpuResults = cpuGrid.batchInterpolate(queryPoints);
//         auto cpuEnd = std::chrono::high_resolution_clock::now();
//         auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
//         std::cout << "CPU interpolation time: " << cpuDuration << " ms" << std::endl;
        
//         // Benchmark GPU interpolation
//         auto gpuStart = std::chrono::high_resolution_clock::now();
//         auto gpuResults = gpuGrid.batchInterpolate(queryPoints);
//         auto gpuEnd = std::chrono::high_resolution_clock::now();
//         auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEnd - gpuStart).count();
//         std::cout << "GPU interpolation time: " << gpuDuration << " ms" << std::endl;
        
//         // Write the expanded interpolated grids to CSV files.
//         writeCSV("cpu_interpolated_grid.csv", cpuResults, new_num_lon, new_num_lat);
//         writeCSV("gpu_interpolated_grid.csv", gpuResults, new_num_lon, new_num_lat);
        
//         std::cout << "Benchmark completed and expanded CSV output generated successfully." << std::endl;
//     } catch (const std::exception& ex) {
//         std::cerr << "Error encountered: " << ex.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }
