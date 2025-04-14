#ifndef GRIDD_H
#define GRIDD_H

#include <vector>
#include <cuda_runtime.h>
#include "Point.h"

// Error checking macro for CUDA calls
#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

/**
 * @brief CUDA implementation of a bathymetry grid with bilinear interpolation
 */
class GridD {
    private:
        double* d_grid;         // Device pointer to the grid data
        int num_lon, num_lat;   // Grid dimensions
        double min_lon, max_lon;  // Longitude range
        double min_lat, max_lat;  // Latitude range
        double lon_step, lat_step;  // Grid cell size
        bool initialized;         // Flag to check if CUDA resources are allocated
    
        /**
         * @brief Initialize CUDA resources
         * 
         * @param elev_data 2D vector of elevation values
         */
        void initialize(const std::vector<std::vector<double>>& elevation_data);
    
    public:
        /**
         * @brief Construct a new CUDA Bathymetry Grid
         * 
         * @param min_longitude Minimum longitude value
         * @param max_longitude Maximum longitude value
         * @param longitude_points Number of grid points in longitude direction
         * @param min_latitude Minimum latitude value
         * @param max_latitude Maximum latitude value
         * @param latitude_points Number of grid points in latitude direction
         * @param elevation_data 2D vector of elevation values
         */
        GridD(double min_longitude, double max_longitude, int longitude_points,
                          double min_latitude, double max_latitude, int latitude_points,
                          const std::vector<std::vector<double>>& elevation_data);
        
        /**
         * @brief Destructor - clean up CUDA resources
         */
        ~GridD();
        
        /**
         * @brief Clean up CUDA resources
         */
        void cleanup();
        
        /**
         * @brief Performs batch bilinear interpolation for multiple points using GPU
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchBilinearInterpolate(const std::vector<Point>& query_points);

        /**
         * @brief Performs batch cubic interpolation for multiple points using GPU
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchCubicInterpolate(const std::vector<Point>& query_points);

        /**
         * @brief Performs batch ordinary kriging interpolation for multiple points using GPU
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchOrdinaryKrigingInterpolate(const std::vector<Point>& query_points);
        
        /**
         * @brief Performs bilinear interpolation at a single point
         * 
         * @param lon Longitude of the query point
         * @param lat Latitude of the query point
         * @return double Interpolated elevation value (NaN if outside grid bounds)
         */
        double bilinearInterpolate(double lon, double lat);
    };

#endif