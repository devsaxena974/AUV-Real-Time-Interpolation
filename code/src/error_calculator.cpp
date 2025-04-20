#include "../include/error_calculator.h"
#include <cmath>
#include <stdexcept>

double meanAbsoluteError(const std::vector<Point>& ref, const std::vector<Point>& interp) {
    if (ref.size() != interp.size() || ref.empty()) {
        throw std::runtime_error("Error: Input point vectors are empty or of different sizes.");
    }
    double sumError = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        // add check to ensure no nan values are involved in calculation
        if(!std::isnan(interp[i].elev)) {
            sumError += fabs(ref[i].elev - interp[i].elev);
        }
    }
    return sumError / ref.size();
}

double rootMeanSquareError(const std::vector<Point>& ref, const std::vector<Point>& interp) {
    if (ref.size() != interp.size() || ref.empty()) {
        throw std::runtime_error("Error: Input point vectors are empty or of different sizes.");
    }
    double sumSq = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        // add check to ensure no nan values are involved in calculation
        if(!std::isnan(interp[i].elev)) {
            double diff = ref[i].elev - interp[i].elev;
            sumSq += diff * diff;
        }
    }
    return sqrt(sumSq / ref.size());
}

double maxAbsoluteError(const std::vector<Point>& ref, const std::vector<Point>& interp) {
    if (ref.size() != interp.size() || ref.empty()) {
        throw std::runtime_error("Error: Input point vectors are empty or of different sizes.");
    }
    double maxErr = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        double diff = fabs(ref[i].elev - interp[i].elev);
        if (diff > maxErr)
            maxErr = diff;
    }
    return maxErr;
}
