import pandas as pd
import numpy as np
import netCDF4 as nc

# -------------------------------
# Step 1: Read the netCDF file and create the grid DataFrame.
# -------------------------------
fn = r'.\GEBCO-Data\GEBCO_28_Feb_2025_5615bda1e072\gebco_2024_n38.2361_s33.7129_w-73.5773_e-70.4713.nc'
ds = nc.Dataset(fn)

lat       = ds['lat'][:]       # 1D array of latitude values
lon       = ds['lon'][:]       # 1D array of longitude values
elevation = ds['elevation'][:] # 2D array of elevations

# Build DataFrame and flip so row=0 is min_lat
data = pd.DataFrame(elevation)
data = data.iloc[::-1].reset_index(drop=True)
rows, cols = data.shape

original_file         = "code/test_data/original_data.csv"      
reduced_file          = "code/test_data/reduced_data.csv"      
reference_file        = "code/test_data/reference_missing.csv"   
reference_coords_file = "code/test_data/reference_missing_coords.csv"

# Save full original grid
data.to_csv(original_file, header=False, index=False)
print(f"Saved original grid to {original_file}.")

# -------------------------------
# Step 2: Randomly select points to remove (no adjacency check)
# -------------------------------
def select_random_points(df, removal_fraction, random_state=42):
    rows, cols = df.shape
    total      = rows * cols
    n_remove   = int(total * removal_fraction)

    np.random.seed(random_state)
    flat = np.random.choice(total, size=n_remove, replace=False)
    return [(idx // cols, idx % cols) for idx in flat]

removal_fraction = 0.01
removed_coords   = select_random_points(data, removal_fraction)
removed_set      = set(removed_coords)
print(f"Selected {len(removed_coords)} points for removal.")

# -------------------------------
# Step 3: Write out the removed points’ grid indices & values
# -------------------------------
idx_rows = []
for r, c in removed_coords:
    idx_rows.append({
        "row":       r,
        "col":       c,
        "elevation": float(data.iat[r, c])
    })
pd.DataFrame(idx_rows).to_csv(reference_file, index=False, header=False)
print(f"Wrote removed row/col reference to {reference_file}.")

# -------------------------------
# Step 4: Write out the removed points’ geographic coords & values
# -------------------------------
coord_rows = []
for r, c in removed_coords:
    coord_rows.append({
        "lon":       float(lon[c]),
        "lat":       float(lat[r]),
        "elevation": float(data.iat[r, c])
    })
pd.DataFrame(coord_rows).to_csv(reference_coords_file, index=False, header=False)
print(f"Wrote removed lon/lat reference to {reference_coords_file}.")

# -------------------------------
# Step 5: Save the reduced grid AS‑IS (no NaNs, still a matrix)
# -------------------------------
# Simply re‑write the original DataFrame—none of its cells were overwritten with NaN.
# data.to_csv(reduced_file, header=False, index=False)
# print(f"Saved reduced grid (grid format) to {reduced_file}.")
df_reduced = data.copy()
for (r, c) in removed_coords:
    df_reduced.iat[r, c] = np.nan

# replace all NaNs with -9999 (or any value you choose)
#df_reduced = df_reduced.fillna(-9999)

df_reduced.to_csv(reduced_file, header=False, index=False, na_rep='nan')
print(f"Saved reduced grid (with removed points = -9999) to {reduced_file}.")
