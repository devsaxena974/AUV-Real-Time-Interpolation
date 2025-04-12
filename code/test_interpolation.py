import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# dataframe_gen(filepath)
#   Input: Filepath string of GEBCO grid data
#   Output: Metadata, dimensions, and pandas DF
def dataframe_gen(filepath):
    ds = nc.Dataset(filepath)

    metadata = ds.__dict__
    dimensions = ds.dimensions
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    elevation = ds['elevation'][:]

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Flatten the arrays
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    elevation_flat = elevation.flatten()

    # Create DataFrame
    df = pd.DataFrame({
        'lat': lat_flat,
        'lon': lon_flat,
        'elevation': elevation_flat
    })

    return metadata, dimensions, df

# interpolate()
# FILL IN REST OF METHOD COMMENTS
def interpolate(
        df:pd.DataFrame,
        lon_flat,
        lat_flat,
        subset_size:float, 
        random_state:int, 
        kriging_model, 
        n_closest_points:int
        ):
    subset_df = df.sample(frac=subset_size, random_state=random_state)

    subset_points = np.array([subset_df['lat'], subset_df['lon']]).T
    subset_elevation = subset_df['elevation']

    grid_points = np.array([lat_flat, lon_flat]).T

    # Cubic interpolation
    cubic_interp_values = griddata(subset_points, subset_elevation, grid_points, method='cubic')
    # Perform bilinear interpolation (linear method in griddata)
    bilinear_interp_values = griddata(subset_points, subset_elevation, grid_points, method='linear')
    # Perform Kriging Interpolation using only the subset
    OK = OrdinaryKriging(
        subset_df['lon'].values, subset_df['lat'].values, subset_df['elevation'].values,  # Input subset
        variogram_model=kriging_model,  # Change to 'spherical', 'gaussian', etc. for tuning
        verbose=False, enable_plotting=False
    )

    # Kriging interpolation on the original full grid
    kriging_interp_values, _ = OK.execute("points", lon_flat, lat_flat, n_closest_points=n_closest_points, backend="loop")

    return subset_df, bilinear_interp_values, cubic_interp_values, kriging_interp_values

def plot_interpolation(df, subset_df, bilinear, cubic, kriging):
    lon_flat = df['lon'].values
    lat_flat = df['lat'].values
    elevation_flat = df['elevation'].values

    plt.rcdefaults()

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    # Original data
    sc1 = axes[0].scatter(lon_flat, lat_flat, c=elevation_flat, s=1, cmap='viridis')
    axes[0].set_title('Original Bathymetry Data')
    fig.colorbar(sc1, ax=axes[0])

    # Subsetted data
    sc2 = axes[1].scatter(subset_df['lon'], subset_df['lat'], c=subset_df['elevation'], s=1, cmap='viridis')
    axes[1].set_title('Subsetted Data (10%)')
    fig.colorbar(sc2, ax=axes[1])

    # Bilinear Interpolated Data
    sc3 = axes[2].scatter(lon_flat, lat_flat, c=bilinear, s=1, cmap='viridis')
    axes[2].set_title('Bilinearly Interpolated Bathymetry Data')
    fig.colorbar(sc3, ax=axes[2])

    # Interpolated data
    sc4 = axes[3].scatter(lon_flat, lat_flat, c=cubic, s=1, cmap='viridis')
    axes[3].set_title('Interpolated Bathymetry Data')
    fig.colorbar(sc4, ax=axes[3])

    # Kriging Interpolated data
    sc5 = axes[4].scatter(lon_flat, lat_flat, c=kriging, s=1, cmap='viridis')
    axes[4].set_title('Kriging Interpolation')
    fig.colorbar(sc5, ax=axes[4])

    plt.tight_layout()
    plt.show()

    return

def calculate_RMSE(df, bilinear, cubic, kriging):
    elevation_range = df['elevation'].values.max() - df['elevation'].values.min()

    valid_mask = ~np.isnan(bilinear)

    bilin_rmse = np.sqrt(mean_squared_error(
        df['elevation'].values[valid_mask], 
        bilinear[valid_mask]
    ))

    bilin_percentage = (bilin_rmse / elevation_range) * 100

    valid_mask = ~np.isnan(cubic)

    cubic_rmse = np.sqrt(mean_squared_error(
        df['elevation'].values[valid_mask], 
        cubic[valid_mask]
    ))

    cubic_percentage = (cubic_rmse / elevation_range) * 100

    valid_mask = ~np.isnan(kriging)

    kriging_rmse = np.sqrt(mean_squared_error(
        df['elevation'].values[valid_mask], 
        kriging[valid_mask]
    ))

    kriging_percentage = (kriging_rmse / elevation_range) * 100

    errors = {
        'bilinear': [bilin_rmse, bilin_percentage],
        'cubic': [cubic_rmse, cubic_percentage],
        'kriging': [kriging_rmse, kriging_percentage]
    }

    return errors