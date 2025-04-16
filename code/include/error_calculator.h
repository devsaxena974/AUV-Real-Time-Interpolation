#ifndef ERROR_CALCULATOR_H
#define ERROR_CALCULATOR_H

#include <vector>
#include "Point.h"

// Computes the Mean Absolute Error between two sets of Points.
double meanAbsoluteError(const std::vector<Point>& ref, const std::vector<Point>& interp);

// Computes the Root Mean Square Error between two sets of Points.
double rootMeanSquareError(const std::vector<Point>& ref, const std::vector<Point>& interp);

// Computes the maximum absolute error between two sets of Points.
double maxAbsoluteError(const std::vector<Point>& ref, const std::vector<Point>& interp);

#endif // ERROR_CALCULATOR_H