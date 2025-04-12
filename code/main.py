from test_interpolation import dataframe_gen, interpolate, plot_interpolation
from test_interpolation import calculate_RMSE

if __name__ == '__main__':
    fp = 'C:\College\EdgeComputing\GEBCO-Data\GEBCO_28_Feb_2025_5615bda1e072\gebco_2024_n38.2361_s33.7129_w-73.5773_e-70.4713.nc'
    meta, dims, df = dataframe_gen(fp)

    subset, bilin, cubic, kriging = interpolate(df,
                                        df['lon'].values,
                                        df['lat'].values,
                                        subset_size=0.01,
                                        random_state=42,
                                        kriging_model='linear',
                                        n_closest_points=50
                                        )
    
    plot_interpolation(df, subset, bilin, cubic, kriging)

    errors = calculate_RMSE(df, bilin, cubic, kriging)
    print(errors)