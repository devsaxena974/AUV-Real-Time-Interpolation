import pandas as pd
import numpy as np
import netCDF4 as nc

# -------------------------------
# Step 1: Read the netCDF file and create the grid DataFrame.
# -------------------------------
fn = r'.\GEBCO-Data\GEBCO_28_Feb_2025_5615bda1e072\gebco_2024_n38.2361_s33.7129_w-73.5773_e-70.4713.nc'
ds = nc.Dataset(fn)

lat = ds['lat'][:]       # 1D array of latitude values
lon = ds['lon'][:]       # 1D array of longitude values
elevation = ds['elevation'][:]

# In this bathymetry application we work directly with the elevation matrix.
data = pd.DataFrame(elevation)

# Invert so that row 0 is min lat not max lat
data = data.iloc[::-1].reset_index(drop=True)

original_file = "code/original_data.csv"      # Full grid matrix
reduced_file = "code/reduced_data.csv"          # Grid with some points masked (NaN)
reference_file = "code/reference_missing.csv"   # Row, col, and original values of removed points
reference_coords_file = "code/reference_missing_coords.csv"  # Lon, lat, and original values of removed points

# Save the full original grid.
data.to_csv(original_file, header=False, index=False)
print(f"Saved original grid (matrix form) to {original_file}.")

# -------------------------------
# Step 2: Select non-adjacent points for removal using an eight-neighborhood test.
# -------------------------------
def select_non_adjacent_points(df, removal_fraction, random_state=42):
    """
    Selects grid points to remove (mask) such that no two removed points are within
    one cell of each other (i.e. not adjacent horizontally, vertically, or diagonally).
    
    Parameters:
        df : pandas.DataFrame
            The grid in matrix form.
        removal_fraction : float
            Fraction of cells to remove.
        random_state : int
            Random seed for reproducibility.
            
    Returns:
        removed : list of tuples (row, col) for the points to be removed.
    """
    rows, cols = df.shape
    total_points = rows * cols
    target_count = int(total_points * removal_fraction)
    
    # Under eight-neighborhood constraints, for a tiled 3x3 block at most one point can be removed.
    # So the theoretical maximum removal is roughly total_points / 9.
    max_possible = total_points // 9 + (1 if total_points % 9 else 0)
    if target_count > max_possible:
        raise ValueError(f"Removal fraction too high with eight-neighborhood constraint: trying to remove {target_count} points but maximum is {max_possible}.")
    
    # Generate a list of all cell coordinates.
    all_coords = [(r, c) for r in range(rows) for c in range(cols)]
    np.random.seed(random_state)
    np.random.shuffle(all_coords)
    
    removed = []
    removed_set = set()
    
    # For each candidate, check its eight surrounding neighbors.
    for (r, c) in all_coords:
        if len(removed) >= target_count:
            break
        conflict = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the candidate itself.
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue  # Ignore out-of-bound neighbors.
                if (nr, nc) in removed_set:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        removed.append((r, c))
        removed_set.add((r, c))
        
    return removed

# Remove, for example, 1% of the points.
removal_fraction = 0.01
removed_coords = select_non_adjacent_points(data, removal_fraction, random_state=42)
print(f"Selected {len(removed_coords)} points to remove with eight-neighborhood exclusion.")

# -------------------------------
# Step 3: Record the reference (row, col, and original elevation) of the removed points.
# -------------------------------
removed_rows = []
for (r, c) in removed_coords:
    removed_rows.append({
        "row": r, 
        "col": c, 
        "elevation": data.iat[r, c]
    })

removed_df = pd.DataFrame(removed_rows)
removed_df.to_csv(reference_file, index=False, header=False)
print(f"Saved reference missing points (row, col) to {reference_file}.")

# -------------------------------
# Step 4: Create a fourth DataFrame with geographic coordinates for the reference missing points.
# Instead of row and column, we use the corresponding lon and lat values.
# -------------------------------
removed_coords_rows = []
for (r, c) in removed_coords:
    removed_coords_rows.append({
        "lon": lon[c],          # Assumes each column corresponds to a longitude in the netCDF file.
        "lat": lat[r],          # Assumes each row corresponds to a latitude in the netCDF file.
        "elevation": data.iat[r, c]
    })

removed_coords_df = pd.DataFrame(removed_coords_rows)
removed_coords_df.to_csv(reference_coords_file, index=False, header=False)
print(f"Saved reference missing points with geographic coordinates to {reference_coords_file}.")

# -------------------------------
# Step 5: Create the masked grid by setting the removed point locations to NaN, and save it.
# -------------------------------
df_reduced = data.copy()
for (r, c) in removed_coords:
    df_reduced.iat[r, c] = np.nan

df_reduced.to_csv(reduced_file, header=False, index=False, na_rep='nan')
print(f"Saved reduced grid (with masked points) to {reduced_file}.")
