#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <string>
#include <array>
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
        std::string gridCSV = "C:/College/EdgeComputing/code/test_data/reduced_data.csv"; // Masked GEBCO grid in matrix format.
        std::string pointsCSV = "C:/College/EdgeComputing/code/test_data/reference_missing.csv"; // Specific points given as row,col,ref_elev.
        std::string outputCPU = "C:/College/EdgeComputing/code/test_data/interpolated_cpu.csv";
        std::string outputGPU = "C:/College/EdgeComputing/code/test_data/interpolated_gpu.csv";

        // open csv file
        std::ofstream resultsCSV("C:/College/EdgeComputing/results/TestingResults1.csv",
            std::ios::out   // write
          | std::ios::app
        );

        if (!resultsCSV.is_open()) {
            throw std::runtime_error("Unable to open TestingResults.csv for writing");
        }

        // Read the masked grid.
        auto gridData = readGridCSV(gridCSV);
        int n_rows = gridData.size();
        if (n_rows == 0)
            throw std::runtime_error("Grid data is empty.");
        int n_cols = gridData[0].size();
        std::cout << "Masked grid dimensions: " << n_cols << " x " << n_rows << std::endl;

        // Define geographic extents (must match how the CSV was generated).
        double min_lon = -73.5773, max_lon = -70.4713;
        double min_lat = 33.7129,  max_lat = 38.2361;
        // Metadata about the grid (must match how the CSV was generated)
        double removalFraction = 0.01;

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
        auto cpu_start1 = std::chrono::high_resolution_clock::now();
        auto interpCPU1 = cpuGrid.batchBilinearInterpolate(queryPoints);
        auto cpu_end1 = std::chrono::high_resolution_clock::now();
        double cpuBilTime1 = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end1 - cpu_start1).count();

        auto cpu_start2 = std::chrono::high_resolution_clock::now();
        auto interpCPU2 = cpuGrid.batchCubicInterpolate(queryPoints);
        auto cpu_end2 = std::chrono::high_resolution_clock::now();
        double cpuCubTime2 = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end2 - cpu_start2).count();

        auto cpu_start3 = std::chrono::high_resolution_clock::now();
        auto interpCPU3 = cpuGrid.batchOrdinaryKrigingInterpolate(queryPoints);
        auto cpu_end3 = std::chrono::high_resolution_clock::now();
        double cpuKrigTime3 = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end3 - cpu_start3).count();
        
        // Run GPU interpolation for these specific points.
        auto gpu_start1 = std::chrono::high_resolution_clock::now();
        auto interpGPU1 = gpuGrid.batchBilinearInterpolate(queryPoints);
        auto gpu_end1 = std::chrono::high_resolution_clock::now();
        double gpuBilTime1 = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end1 - gpu_start1).count();

        auto gpu_start2 = std::chrono::high_resolution_clock::now();
        auto interpGPU2 = gpuGrid.batchCubicInterpolate(queryPoints);
        auto gpu_end2 = std::chrono::high_resolution_clock::now();
        double gpuCubTime2 = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end2 - gpu_start2).count();

        auto gpu_start3 = std::chrono::high_resolution_clock::now();
        auto interpGPU3 = gpuGrid.batchOrdinaryKrigingInterpolate(queryPoints);
        auto gpu_end3 = std::chrono::high_resolution_clock::now();
        double gpuKrigTime3 = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end3 - gpu_start3).count();

        // Write the results (point list format) to CSV.
        writePointsCSV(outputCPU, interpCPU3);
        writePointsCSV(outputGPU, interpGPU3);
        std::cout << "Bilinear Interpolated CPU results written to " << outputCPU << std::endl;
        std::cout << "Bilinear Interpolated GPU results written to " << outputGPU << std::endl;

        // Print timings
        std::cout << "  CPU Bilinear: " << cpuBilTime1 << " ms" << std::endl;
        std::cout << "  GPU Bilinear: " << gpuBilTime1 << " ms" << std::endl;
        std::cout << "  CPU Cubic: " << cpuCubTime2 << " ms" << std::endl;
        std::cout << "  GPU Cubic: " << gpuCubTime2 << " ms" << std::endl;
        std::cout << "  CPU Kriging: " << cpuKrigTime3 << " ms" << std::endl;
        std::cout << "  GPU Kriging: " << gpuKrigTime3 << " ms" << std::endl;

        // Compute error metrics for bilinear.
        double maeCPU = meanAbsoluteError(refValues, interpCPU1);
        double rmseCPU = rootMeanSquareError(refValues, interpCPU1);
        double maxErrCPU = maxAbsoluteError(refValues, interpCPU1);
        double maeGPU = meanAbsoluteError(refValues, interpGPU1);
        double rmseGPU = rootMeanSquareError(refValues, interpGPU1);
        double maxErrGPU = maxAbsoluteError(refValues, interpGPU1);



        std::cout << "\nCPU Bilinear Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeCPU << std::endl;
        std::cout << "  RMSE = " << rmseCPU << std::endl;
        std::cout << "  Max  = " << maxErrCPU << std::endl;
        std::cout << "\nGPU Bilinear Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeGPU << std::endl;
        std::cout << "  RMSE = " << rmseGPU << std::endl;
        std::cout << "  Max  = " << maxErrGPU << std::endl;

        // Compute error metrics for cubic
        double maeCPU1 = meanAbsoluteError(refValues, interpCPU2);
        double rmseCPU1 = rootMeanSquareError(refValues, interpCPU2);
        double maxErrCPU1 = maxAbsoluteError(refValues, interpCPU2);
        double maeGPU1 = meanAbsoluteError(refValues, interpGPU2);
        double rmseGPU1 = rootMeanSquareError(refValues, interpGPU2);
        double maxErrGPU1 = maxAbsoluteError(refValues, interpGPU2);

        std::cout << "\nCPU Cubic Splines Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeCPU1 << std::endl;
        std::cout << "  RMSE = " << rmseCPU1 << std::endl;
        std::cout << "  Max  = " << maxErrCPU1 << std::endl;
        std::cout << "\nGPU Cubic Splines Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeGPU1 << std::endl;
        std::cout << "  RMSE = " << rmseGPU1 << std::endl;
        std::cout << "  Max  = " << maxErrGPU1 << std::endl;

        // Compute error metrics for kriging
        double maeCPU2 = meanAbsoluteError(refValues, interpCPU3);
        double rmseCPU2 = rootMeanSquareError(refValues, interpCPU3);
        double maxErrCPU2 = maxAbsoluteError(refValues, interpCPU3);
        double maeGPU2 = meanAbsoluteError(refValues, interpGPU3);
        double rmseGPU2 = rootMeanSquareError(refValues, interpGPU3);
        double maxErrGPU2 = maxAbsoluteError(refValues, interpGPU3);

        std::cout << "\nCPU Ordinary Kriging Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeCPU2 << std::endl;
        std::cout << "  RMSE = " << rmseCPU2 << std::endl;
        std::cout << "  Max  = " << maxErrCPU2 << std::endl;
        std::cout << "\nGPU Ordinary Kriging Interpolation Errors:" << std::endl;
        std::cout << "  MAE  = " << maeGPU2 << std::endl;
        std::cout << "  RMSE = " << rmseGPU2 << std::endl;
        std::cout << "  Max  = " << maxErrGPU2 << std::endl;

        // Define a little struct to hold each test’s metadata+metrics:
        struct TestResult {
            const char* machine;          // "CPU" or "GPU"
            const char* method;           // "Bilinear", "Cubic" or "Kriging"
            double      time_ms;          // the timing you measured
            double      mae, rmse, maxErr;// the three error metrics
        };

        std::array<TestResult,6> allResults = { {
            { "CPU", "Bilinear", cpuBilTime1,  maeCPU,  rmseCPU,  maxErrCPU  },
            { "CPU", "Cubic",    cpuCubTime2,  maeCPU1, rmseCPU1, maxErrCPU1 },
            { "CPU", "Kriging",  cpuKrigTime3, maeCPU2, rmseCPU2, maxErrCPU2 },
            { "GPU", "Bilinear", gpuBilTime1,  maeGPU,  rmseGPU,  maxErrGPU  },
            { "GPU", "Cubic",    gpuCubTime2,  maeGPU1, rmseGPU1, maxErrGPU1 },
            { "GPU", "Kriging",  gpuKrigTime3, maeGPU2, rmseGPU2, maxErrGPU2 }
        } };

        // append to csv file
        for(auto &r : allResults) {
            resultsCSV
                << r.machine           << ','   // Machine
                << r.method            << ','   // InterpolationType
                << "B"                 << ','   // GridType (B for “missing‐point” test)
                << pointIndices.size() << ','   // BatchSize
                << r.time_ms           << ','   // Time (ms)
                << removalFraction     << ','   // RemovalFraction
                << r.mae               << ','   // MAE
                << r.rmse              << ','   // RMSE
                << r.maxErr                     // MaxError
                << '\n';

            std::cout << "Wrote the following to csv: " << r.machine << " " << r.method << std::endl;
        }
        
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
