# A Study of Real Time Interpolation for Deep Sea AUV Mapping

This repository contains the code required to generate, test, and analyze the performace
differences between CPU and GPU implementations of common spatial interpolation methods used
in seafloor mapping.

## File Structure

- GEBCO-Data
    - Contains data obtained from https://download.gebco.net/ to test real-life scenarios
- code
    - Contains CPU & GPU implementation, test code, grid generation code, and error calculation code
- results
    - Contains CSV file with results on each test run
- writing
    - Contains PDFs and Powerpoints I used to present my work

## Data & Usage

The code in this repository is designed to run on 2 different grid types:
- Grid A: Generated with code/generate_csv_grids.cpp
    - Has fully populated longitudes for each latitude based on provided bounds
    - Interpolation needed to increase frequency/resolution of grid by generating points between each pre-existing latitude
    - Tested in code/test_interpolation.cpp

- Grid B: Uses existing GEBCO-Data that is randomly masked with a set removal fraction in code/subset_bathymetry.py
    - Grids with artifically removed points are stored in code/test_data folder
    - Interpolation needed to fill in missing points and replicate scenarios where points have been augmented, there is noise, or hardware issues cause faulty sonar recordings
    - Tested in code/test_gebco.cpp

## Recording Results

- Most of the performance and accuracy metrics are stored in results/TestingResults1.csv upon running of the test files
- Graphs are generated in code/graph_results.ipynb
- Specific performance & accuracy metrics are found in performance_results.ipynb
    - NOTE: For Grid B testing, the performance and accuracy metrics recording system needs to be changed to account for what region the data comes from

For more information and details on the testing and math involved, see the 'writing' folder for the full paper


