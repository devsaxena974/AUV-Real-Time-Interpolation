#ifndef GRIDH_H
#define GRIDH_H

#include <vector>
#include "Point.h"

class GridH {
    private:
        // Store all data points in a grid
        std::vector<std::vector<double>> elevations;

        double max_lat, min_lat;
        double max_lon, min_lon;

        int num_lat, num_lon;     // Grid dimensions

        double lat_step, lon_step;  // Grid cell size

    public:
        GridH(double max_lat,
            double min_lat,
            int n_lat_points,
            double max_lon,
            double min_lon,
            int n_lon_points,
            std::vector<std::vector<double>>& elevation_data
        );

        /**
         * @brief Performs bilinear interpolation at a single point
         * 
         * @param lon Longitude of the query point
         * @param lat Latitude of the query point
         * @return double Interpolated elevation value (NaN if outside grid bounds)
         */
        double bilinearInterpolate(double lon, double lat) const;

        /**
         * @brief Performs cubic spline interpolation at a single point
         * 
         * @param lon Longitude of the query point
         * @param lat Latitude of the query point
         * @return double Interpolated elevation value (NaN if outside grid bounds)
         */
        double cubicInterpolate(double lon, double lat) const;

        /**
         * @brief Performs ordinary kriging interpolation at a single point
         * 
         * @param lon Longitude of the query point
         * @param lat Latitude of the query point
         * @return double Interpolated elevation value (NaN if outside grid bounds)
         */
        double ordinaryKrigingInterpolate(double lon, double lat) const;

        /**
         * @brief Performs batch bilinear interpolation for multiple points
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchBilinearInterpolate(const std::vector<Point>& query_points) const;

        /**
         * @brief Performs batch cubic interpolation for multiple points
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchCubicInterpolate(const std::vector<Point>& query_points) const;

        /**
         * @brief Performs batch kriging interpolation for multiple points
         * 
         * @param query_points Vector of points to interpolate
         * @return std::vector<Point> Interpolated results
         */
        std::vector<Point> batchOrdinaryKrigingInterpolate(const std::vector<Point>& query_points) const;

};

#endif