#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>

// Include your grid and point classes – ensure these header files are in your include path.
#include "include/Point.h"
#include "include/GridH.h"
#include "include/GridD.h"

// ----------------------------------------------------------------
// Helper Function: Read CSV file into a 2D vector of doubles.
// ----------------------------------------------------------------
std::vector<std::vector<double>> readCSV(const std::string &filename) {
    std::vector<std::vector<double>> grid;
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Unable to open CSV file: " + filename);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        grid.push_back(row);
    }
    return grid;
}

// ----------------------------------------------------------------
// Helper Function: Generate random test points uniformly distributed
// within the provided domain.
// ----------------------------------------------------------------
std::vector<Point> generateTestPoints(int num_points, double min_lon, double max_lon, double min_lat, double max_lat) {
    std::vector<Point> points(num_points);
    std::random_device rd;
    std::mt19937 gen(rd());
    const double epsilon = 1e-6;
    std::uniform_real_distribution<> lon_dist(min_lon + epsilon, max_lon - epsilon);
    std::uniform_real_distribution<> lat_dist(min_lat + epsilon, max_lat - epsilon);

    for (int i = 0; i < num_points; ++i)
        points[i] = {lon_dist(gen), lat_dist(gen), 0.0};

    return points;
}

// ----------------------------------------------------------------
// Helper Function: Write a 2D grid of Points (in row-major order)
// to a CSV file. The grid dimensions (cols x rows) must be provided.
// ----------------------------------------------------------------
void writeCSV(const std::string &filename, const std::vector<Point> &points,
        int gridCols, int gridRows) {
    std::ofstream file(filename);
    if (!file.is_open())
    throw std::runtime_error("Unable to open file for writing: " + filename);
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

// ----------------------------------------------------------------
// Helper Function: Generate an expanded grid of query points that
// covers the same geographic extents but with increased resolution.
// The new grid dimensions are defined as:
//    new_num_lon = 2 * num_lon - 1
//    new_num_lat = 2 * num_lat - 1
//
// This grid will contain both the original points (at even indices)
// and the new interpolated points (at positions where at least one index
// is odd).
// ----------------------------------------------------------------
std::vector<Point> generateExpandedGridQueryPoints(int num_lon, int num_lat,
        double min_lon, double max_lon,
        double min_lat, double max_lat) {
    int new_num_lon = 2 * num_lon - 1;
    int new_num_lat = 2 * num_lat - 1;
    std::vector<Point> points;
    points.reserve(new_num_lon * new_num_lat);

    for (int j = 0; j < new_num_lat; ++j) {
        // Linear interpolation for latitude
        double lat = min_lat + j * (max_lat - min_lat) / (new_num_lat - 1);
        for (int i = 0; i < new_num_lon; ++i) {
            // Linear interpolation for longitude
            double lon = min_lon + i * (max_lon - min_lon) / (new_num_lon - 1);
            points.push_back({lon, lat, 0.0});
        }
    }
    return points;
}

// ----------------------------------------------------------------
// Main Function: Hardcoded to use one CSV file ("grid_medium.csv")
// and running multiple batch interpolation tests.
// ----------------------------------------------------------------
int main() {
    try {
        // Print current working directory (useful for debugging file paths)
        //std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

        // Hardcode the CSV file name.
        std::string csvFilename = "grid_large.csv";
        std::cout << "Using CSV file: " << csvFilename << std::endl;

        // Read raw grid data from the CSV file.
        auto rawGrid = readCSV(csvFilename);
        int num_lat = rawGrid.size();
        if (num_lat == 0)
            throw std::runtime_error("CSV grid data is empty!");
        int num_lon = rawGrid[0].size();
        std::cout << "CSV grid dimensions: " << num_lon << " x " << num_lat << std::endl;

        // Set geographic extents (these must match how the CSV was generated).
        double min_lon = -180.0, max_lon = -160.0;
        double min_lat = 20.0,  max_lat = 30.0;

        // Create CPU and GPU grid objects.
        // GridH constructor: (max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, elevation_data)
        GridH cpuGrid(max_lat, min_lat, num_lat, max_lon, min_lon, num_lon, rawGrid);
        // GridD constructor: (min_lon, max_lon, num_lon, min_lat, max_lat, num_lat, elevation_data)
        GridD gpuGrid(min_lon, max_lon, num_lon, min_lat, max_lat, num_lat, rawGrid);

        // List of batch sizes to test (for random query points).
        std::vector<int> batch_sizes = {1000, 5000, 10000, 50000, 100000, 1000000, 5000000};

        std::cout << "\nStarting CPU vs. GPU interpolation benchmarks:" << std::endl;
        for (const auto &batch_size : batch_sizes) {
            std::cout << "\nTesting with " << batch_size << " random points:" << std::endl;
            auto testPoints = generateTestPoints(batch_size, min_lon, max_lon, min_lat, max_lat);
            
            // CPU bilinear interpolation timing.
            auto cpu_start = std::chrono::high_resolution_clock::now();
            auto cpuBilinear = cpuGrid.batchBilinearInterpolate(testPoints);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            double cpuBilTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
            
            // CPU cubic interpolation timing.
            cpu_start = std::chrono::high_resolution_clock::now();
            auto cpuCubic = cpuGrid.batchCubicInterpolate(testPoints);
            cpu_end = std::chrono::high_resolution_clock::now();
            double cpuCubTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

            // CPU kriging interpolation timing.
            cpu_start = std::chrono::high_resolution_clock::now();
            auto cpuKriging = cpuGrid.batchOrdinaryKrigingInterpolate(testPoints);
            cpu_end = std::chrono::high_resolution_clock::now();
            double cpuKrigTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
            
            // GPU bilinear interpolation timing.
            auto gpu_start = std::chrono::high_resolution_clock::now();
            auto gpuBilinear = gpuGrid.batchBilinearInterpolate(testPoints);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            double gpuBilTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
            
            // GPU cubic interpolation timing.
            gpu_start = std::chrono::high_resolution_clock::now();
            auto gpuCubic = gpuGrid.batchCubicInterpolate(testPoints);
            gpu_end = std::chrono::high_resolution_clock::now();
            double gpuCubTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();

            // GPU kriging interpolation timing.
            gpu_start = std::chrono::high_resolution_clock::now();
            auto gpuKriging = gpuGrid.batchOrdinaryKrigingInterpolate(testPoints);
            gpu_end = std::chrono::high_resolution_clock::now();
            double gpuKrigTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
            
            std::cout << "  CPU Bilinear: " << cpuBilTime << " ms" << std::endl;
            std::cout << "  CPU Cubic: " << cpuCubTime << " ms" << std::endl;
            std::cout << "  CPU Kriging: " << cpuKrigTime << " ms" << std::endl;
            std::cout << "  GPU Bilinear: " << gpuBilTime << " ms" << std::endl;
            std::cout << "  GPU Cubic: " << gpuCubTime << " ms" << std::endl;
            std::cout << "  GPU Kriging: " << gpuKrigTime << " ms" << std::endl;
            
            // Validate a few sample interpolation results.
            int check_count = std::min(10, batch_size);
            bool valid = true;
            for (int i = 0; i < check_count; ++i) {
                double diff = std::abs(cpuBilinear[i].elev - gpuBilinear[i].elev);
                if(diff > 1e-6) {
                    valid = false;
                    std::cout << "  Bilinear mismatch at point " << i << ": CPU = " 
                              << cpuBilinear[i].elev << ", GPU = " << gpuBilinear[i].elev << std::endl;
                    break;
                }
            }
            std::cout << "  Bilinear result validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
            
            valid = true;
            for (int i = 0; i < check_count; ++i) {
                double diff = std::abs(cpuCubic[i].elev - gpuCubic[i].elev);
                if(diff > 1e-6) {
                    valid = false;
                    std::cout << "  Cubic mismatch at point " << i << ": CPU = " 
                              << cpuCubic[i].elev << ", GPU = " << gpuCubic[i].elev << std::endl;
                    break;
                }
            }
            std::cout << "  Cubic result validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

            valid = true;
            for (int i = 0; i < check_count; ++i) {
                double diff = std::abs(cpuKriging[i].elev - gpuKriging[i].elev);
                if(diff > 1e-6) {
                    valid = false;
                    std::cout << "  Kriging mismatch at point " << i << ": CPU = " 
                              << cpuKriging[i].elev << ", GPU = " << gpuKriging[i].elev << std::endl;
                    break;
                }
            }
            std::cout << "  Kriging result validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
        }
        
        // ----------------------------------------------------------------
        // Generate the Expanded Grid CSV for each interpolation method.
        // The expanded grid dimensions:
        //     new_num_lon = 2 * num_lon - 1
        //     new_num_lat = 2 * num_lat - 1
        int new_num_lon = 2 * num_lon - 1;
        int new_num_lat = 2 * num_lat - 1;
        
        // Generate expanded grid query points.
        auto expandedQueryPoints = generateExpandedGridQueryPoints(num_lon, num_lat, min_lon, max_lon, min_lat, max_lat);
        
        // Use CPU grid to interpolate on the expanded grid for both bilinear and cubic.
        auto expanded_cpu_bilinear = cpuGrid.batchBilinearInterpolate(expandedQueryPoints);
        auto expanded_cpu_cubic = cpuGrid.batchCubicInterpolate(expandedQueryPoints);
        auto expanded_cpu_kriging = cpuGrid.batchOrdinaryKrigingInterpolate(expandedQueryPoints);
        
        // Use GPU grid to interpolate on the expanded grid for both bilinear and cubic.
        auto expanded_gpu_bilinear = gpuGrid.batchBilinearInterpolate(expandedQueryPoints);
        auto expanded_gpu_cubic = gpuGrid.batchCubicInterpolate(expandedQueryPoints);
        auto expanded_gpu_kriging = gpuGrid.batchOrdinaryKrigingInterpolate(expandedQueryPoints);
        
        // Write each expanded grid to a separate CSV file.
        // writeCSV("expanded_cpu_bilinear_grid.csv", expanded_cpu_bilinear, new_num_lon, new_num_lat);
        // writeCSV("expanded_cpu_cubic_grid.csv", expanded_cpu_cubic, new_num_lon, new_num_lat);
        // writeCSV("expanded_cpu_kriging_grid.csv", expanded_cpu_kriging, new_num_lon, new_num_lat);
        // writeCSV("expanded_gpu_bilinear_grid.csv", expanded_gpu_bilinear, new_num_lon, new_num_lat);
        // writeCSV("expanded_gpu_cubic_grid.csv", expanded_gpu_cubic, new_num_lon, new_num_lat);
        // writeCSV("expanded_gpu_kriging_grid.csv", expanded_gpu_kriging, new_num_lon, new_num_lat);
        
        // std::cout << "\nExpanded interpolated grid CSVs generated:" << std::endl;
        // std::cout << "  expanded_cpu_bilinear_grid.csv" << std::endl;
        // std::cout << "  expanded_cpu_cubic_grid.csv" << std::endl;
        // std::cout << "  expanded_cpu_kriging_grid.csv" << std::endl;
        // std::cout << "  expanded_gpu_bilinear_grid.csv" << std::endl;
        // std::cout << "  expanded_gpu_cubic_grid.csv" << std::endl;
        // std::cout << "  expanded_gpu_kriging_grid.csv" << std::endl;
        
        std::cout << "\nBenchmarking complete." << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
