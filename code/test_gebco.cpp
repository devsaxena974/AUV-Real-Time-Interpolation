#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <string>
#include "include/Point.h"
#include "include/GridH.h"
#include "include/GridD.h"
#include "include/error_calculator.h"  // Functions: meanAbsoluteError, rootMeanSquareError, maxAbsoluteError

// ----------------------------------------------------------------
// Helper Function: Read a grid CSV (matrix format) into a 2D vector of doubles.
// Each row in the CSV corresponds to one latitude and each column to one longitude.
std::vector<std::vector<double>> readGridCSV(const std::string &filename) {
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
            if (!cell.empty())
                row.push_back(std::stod(cell));
        }
        // Ensure that even rows that might be entirely "nan" or blank become full rows.
        if (row.empty() && !grid.empty()) {
            row.resize(grid[0].size(), std::numeric_limits<double>::quiet_NaN());
        }
        grid.push_back(row);
    }
    return grid;
}

// ----------------------------------------------------------------
// Updated Helper Function: Read specific query points from a CSV file.
// Now, each row in the CSV contains: row, col, ref_elev
// (These are grid indices, not geographic coordinates.)
// We'll later convert these indices to lon/lat.
std::vector<std::tuple<int, int, double>> readSpecificPointIndices(const std::string &filename) {
    std::vector<std::tuple<int, int, double>> points;
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Unable to open specific points CSV: " + filename);
    std::string line;
    // Uncomment below if a header exists:
    // std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        int row, col;
        double ref;
        char comma;
        if (ss >> row >> comma >> col >> comma >> ref) {
            points.push_back(std::make_tuple(row, col, ref));
        }
    }
    return points;
}

// ----------------------------------------------------------------
// Helper Function: Convert grid indices (row, col) to geographic coordinates.
// Uses totalRows and totalCols from the full (original or masked) grid.

void gridIndexToGeo(int row, int col, int totalRows, int totalCols,
        double min_lat, double max_lat,
        double min_lon, double max_lon,
        double &lat, double &lon) {
    double lat_step = (max_lat - min_lat) / (totalRows - 1);
    double lon_step = (max_lon - min_lon) / (totalCols - 1);
    // With data reordered (row 0 = min_lat), simply:
    lat = min_lat + row * lat_step;
    lon = min_lon + col * lon_step;
}

// ----------------------------------------------------------------
// Helper Function: Write a vector of Points (point list format) to CSV.
// Each row: lon,lat,interpolated_value.
void writePointsCSV(const std::string &filename, const std::vector<Point> &points) {
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Unable to open file for writing: " + filename);
    // Optionally write header.
    file << "lon,lat,interpolated_value\n";
    for (const auto &pt : points) {
        file << pt.lon << "," << pt.lat << "," << pt.elev << "\n";
    }
    file.close();
}

// ----------------------------------------------------------------
// Main Test Function: Use specific points (given as grid indices) for interpolation.
int main() {
    try {
        // File names:
        std::string gridCSV = "C:/College/EdgeComputing/code/reduced_data.csv"; // Masked GEBCO grid in matrix format.
        std::string pointsCSV = "C:/College/EdgeComputing/code/reference_missing.csv"; // Specific points given as row,col,ref_elev.
        std::string outputCPU = "C:/College/EdgeComputing/code/interpolated_cpu.csv";
        std::string outputGPU = "C:/College/EdgeComputing/code/interpolated_gpu.csv";

        // Read the masked grid.
        auto gridData = readGridCSV(gridCSV);
        int n_rows = gridData.size();
        if (n_rows == 0)
            throw std::runtime_error("Grid data is empty.");
        int n_cols = gridData[0].size();
        std::cout << "Masked grid dimensions: " << n_cols << " x " << n_rows << std::endl;

        // Define geographic extents (must match how the CSV was generated).
        double min_lon = -73.57708333333333, max_lon = -70.47291666666666;
        double min_lat = 33.71458333333334,  max_lat = 38.23541666666665;

        // Create grid objects.
        // GridH: (max_lat, min_lat, n_rows, max_lon, min_lon, n_cols, gridData)
        GridH cpuGrid(max_lat, min_lat, n_rows, max_lon, min_lon, n_cols, gridData);
        // GridD: (min_lon, max_lon, n_cols, min_lat, max_lat, n_rows, gridData)
        GridD gpuGrid(min_lon, max_lon, n_cols, min_lat, max_lat, n_rows, gridData);

        // Read specific points given as grid indices (row, col, original_value).
        auto pointIndices = readSpecificPointIndices(pointsCSV);
        if (pointIndices.empty())
            throw std::runtime_error("No specific query points found in " + pointsCSV);
        std::cout << "Read " << pointIndices.size() << " specific points (grid indices)." << std::endl;

        // For each read point, convert (row, col) to geographic coordinates using the full grid dims.
        std::vector<Point> queryPoints;
        std::vector<Point> refValues;
        for (const auto &t : pointIndices) {
            int r, c;
            double orig;
            std::tie(r, c, orig) = t;
            double lat, lon;
            gridIndexToGeo(r, c, n_rows, n_cols, min_lat, max_lat, min_lon, max_lon, lat, lon);
            queryPoints.push_back({lon, lat, 0.0});
            refValues.push_back({lon, lat, orig});
        }
        // DEBUGGING: PRINT FIRST 5 queryPoints
        for(int i  = 0; i < 5; i++) {
            std::cout << queryPoints[i].lon << " " << queryPoints[i].lat << " " << queryPoints[i].elev<< std::endl;
        }

        // Run CPU interpolation for these specific points.
        auto cpu_start = std::chrono::high_resolution_clock::now();
        //auto interpCPU = cpuGrid.batchBilinearInterpolate(queryPoints);
        //auto interpCPU = cpuGrid.batchCubicInterpolate(queryPoints);
        auto interpCPU = cpuGrid.batchOrdinaryKrigingInterpolate(queryPoints);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpuBilTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
        // Run GPU interpolation for these specific points.
        auto gpu_start = std::chrono::high_resolution_clock::now();
        //auto interpGPU = gpuGrid.batchBilinearInterpolate(queryPoints);
        //auto interpGPU = gpuGrid.batchCubicInterpolate(queryPoints);
        auto interpGPU = gpuGrid.batchOrdinaryKrigingInterpolate(queryPoints);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        double gpuBilTime = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();

        // Write the results (point list format) to CSV.
        writePointsCSV(outputCPU, interpCPU);
        writePointsCSV(outputGPU, interpGPU);
        std::cout << "Interpolated CPU results written to " << outputCPU << std::endl;
        std::cout << "Interpolated GPU results written to " << outputGPU << std::endl;

        // Print timings
        std::cout << "  CPU Bilinear: " << cpuBilTime << " ms" << std::endl;
        std::cout << "  GPU Bilinear: " << gpuBilTime << " ms" << std::endl;

        // Compute error metrics.
        double maeCPU = meanAbsoluteError(refValues, interpCPU);
        double rmseCPU = rootMeanSquareError(refValues, interpCPU);
        double maxErrCPU = maxAbsoluteError(refValues, interpCPU);
        double maeGPU = meanAbsoluteError(refValues, interpGPU);
        double rmseGPU = rootMeanSquareError(refValues, interpGPU);
        double maxErrGPU = maxAbsoluteError(refValues, interpGPU);

        std::cout << "\nCPU Bilinear Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeCPU << std::endl;
        std::cout << "  RMSE = " << rmseCPU << std::endl;
        std::cout << "  Max  = " << maxErrCPU << std::endl;
        std::cout << "\nGPU Bilinear Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeGPU << std::endl;
        std::cout << "  RMSE = " << rmseGPU << std::endl;
        std::cout << "  Max  = " << maxErrGPU << std::endl;
        
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
