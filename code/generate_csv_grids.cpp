#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <sstream>

// // Generate a test grid using the formula:
// //    elevations[j][i] = -1000.0 - 10.0 * sin(i * 0.01) - 15.0 * cos(j * 0.01);
// std::vector<std::vector<double>> generateTestGrid(int num_lon, int num_lat) {
//     std::vector<std::vector<double>> grid(num_lat, std::vector<double>(num_lon));
//     for (int j = 0; j < num_lat; ++j) {
//         for (int i = 0; i < num_lon; ++i) {
//             grid[j][i] = -1000.0 - 10.0 * sin(i * 0.01) - 15.0 * cos(j * 0.01);
//         }
//     }
//     return grid;
// }
// Generate a test grid using a more realistic bathymetry equation.
// The grid is generated in a 2D domain where:
//  - x (offshore) ranges from 0 to L (with 0 at the coast)
//  - y (alongshore) ranges from 0 to W
// The depth is computed as:
//     depth(x,y) = - (d0 + k * x) + A * exp( - [ (x - x_m)^2/(2*sigma_x^2) + (y - y_m)^2/(2*sigma_y^2) ] )
// Here:
//  d0: nearshore depth magnitude (e.g. 10 m)
//  k: rate of deepening offshore (e.g., 2 m per unit distance)
//  A: amplitude of the underwater mountain (100 m)
//  (x_m, y_m): mountain center (e.g., x=75, y=50)
//  (sigma_x, sigma_y): mountain spread (e.g., 15 each)
// For points far from the mountain, the exponential term decays, and the baseline deepening dominates.
std::vector<std::vector<double>> generateTestGrid(int num_lon, int num_lat) {
    std::vector<std::vector<double>> grid(num_lat, std::vector<double>(num_lon));
    
    // Set the physical extents for the grid.
    // x: offshore direction (0 = coast, L = offshore limit)
    // y: alongshore direction
    double L = 100.0;  // offshore distance (units can be kilometers or arbitrary)
    double W = 100.0;  // alongshore width

    // Parameters for the depth equation:
    double d0 = 10.0;   // nearshore depth magnitude (e.g., at the coast, depth = -10 m)
    double k  = 2.0;    // rate of deepening: each unit in x adds 2 m of depth
    double A  = 100.0;  // amplitude of the underwater mountain (shallower bump)
    double x_m = 75.0;  // mountain center offshore (toward the right side)
    double y_m = 50.0;  // mountain center alongshore (middle of the domain)
    double sigma_x = 15.0;  // mountain horizontal spread
    double sigma_y = 15.0;  // mountain vertical spread

    // Generate the grid by mapping i->x and j->y.
    for (int j = 0; j < num_lat; ++j) {
        // Map row index to y coordinate in [0, W]
        double y = W * j / (num_lat - 1);
        for (int i = 0; i < num_lon; ++i) {
            // Map column index to x coordinate in [0, L]
            double x = L * i / (num_lon - 1);

            // Baseline offshore slope: depth becomes more negative with increasing x.
            double baseline = -(d0 + k * x);

            // Gaussian underwater mountain centered at (x_m, y_m).
            double mountain = A * exp( - ( (x - x_m) * (x - x_m) / (2 * sigma_x * sigma_x)
                                           + (y - y_m) * (y - y_m) / (2 * sigma_y * sigma_y) ) );

            // The final depth is the sum of the baseline and the mountain.
            grid[j][i] = baseline + mountain;
        }
    }
    return grid;
}

// Write a 2D grid (vector of vectors) to a CSV file.
void writeCSVGrid(const std::string& filename, const std::vector<std::vector<double>>& grid) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    for (const auto &row : grid) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

int main() {
    try {
        // Define several grid sizes (you can adjust or add more)
        struct GridSize {
            int num_lon;
            int num_lat;
            std::string filename;
        };
        
        std::vector<GridSize> sizes = {
            //{10, 8, "grid_small.csv"}
            // {1000, 800, "grid_medium.csv"},
            //{2000, 1600, "grid_large.csv"},
            {4000, 3200, "grid_large.csv"}
        };
        
        for (const auto &size : sizes) {
            auto grid = generateTestGrid(size.num_lon, size.num_lat);
            writeCSVGrid(size.filename, grid);
            std::cout << "Generated CSV: " << size.filename 
                      << " (Dimensions: " << size.num_lon << " x " << size.num_lat << ")" 
                      << std::endl;
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
