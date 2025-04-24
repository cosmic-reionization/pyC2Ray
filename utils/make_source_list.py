# generates a source list to be used with the source models in the pyc2ray_ramses class
# the source strengths will have 3 columns LMACHs, HMACHs, Mass-dependent_LMACH

import pandas as pd
import numpy as np
import sys
import glob
import re

start_snap = 14
end_snap = 45


# Define the size of the mesh
N = 256
h = 0.677
HMACH = (1.43e9) / h  # in solar masses
boxsize = 100  # Mpc
AHF_path = '/store/clues/HESTIA/RE_SIMS/4096/DM_ONLY/37_11/AHF_output/'

for i in range(start_snap,end_snap+1):
    
    AHF_directory = '{}/HESTIA_100Mpc_4096_37_11.{:03d}.*.AHF_halos'.format(AHF_path, i)

    # Get list of files ending with "_halos"
    halos_files = glob.glob(AHF_directory)
    # Print full file paths
    for AHF_file in halos_files:
        print(AHF_file)

    # Regular expression pattern to match the number between 'z' and '.AHF'
    pattern = r'z(\d+\.\d+)'

    # Use re.search to find the match
    match = re.search(pattern, AHF_file)

    # Extract the number from the match
    if match:
        redshift = match.group(1)
        print(redshift)
        z  = float(redshift)

    # Prepare the output grid: N^3 cells, 3 fields (HMACH, LMACH, LMACH_MassDep)
    HMACH_grid = np.zeros((N, N, N))
    LMACH_grid = np.zeros((N, N, N))
    LMACH_MassDep_grid = np.zeros((N, N, N))


    print('Doing z=',redshift)

    # Read the halo catalog
    df = pd.read_csv(AHF_file, delim_whitespace=True)
    df = df[df['Mvir(4)']/h>10**8]
    # Extract position and mass data
    x = np.array(df['Xc(6)']) / 1000.0  # Convert to Mpc
    y = np.array(df['Yc(7)']) / 1000.0
    z = np.array(df['Zc(8)']) / 1000.0
    mass = np.array(df['Mvir(4)']) / h  # Mass in solar masses

    # Convert positions to grid indices
    grid_x = np.floor((x / boxsize) * N).astype(int)
    grid_y = np.floor((y / boxsize) * N).astype(int)
    grid_z = np.floor((z / boxsize) * N).astype(int)

    # Ensure that grid indices stay within bounds
    grid_x = np.clip(grid_x, 0, N - 1)
    grid_y = np.clip(grid_y, 0, N - 1)
    grid_z = np.clip(grid_z, 0, N - 1)

    # Update grid mass values
    for j in range(len(mass)):
        if mass[j] >= HMACH:
            HMACH_grid[grid_x[j], grid_y[j], grid_z[j]] += mass[j]
        else:
            LMACH_grid[grid_x[j], grid_y[j], grid_z[j]] += mass[j]
            supp_factor = ((mass[j] / (9 * 10**8)) - 1/9)
            if supp_factor >= 0:
                LMACH_MassDep_grid[grid_x[j], grid_y[j], grid_z[j]] += supp_factor * mass[j]

    # Convert grid data into DataFrame for output
    src = pd.DataFrame({
        'x': np.repeat(np.arange(1, N + 1), N * N),
        'y': np.tile(np.repeat(np.arange(1, N + 1), N), N),
        'z': np.tile(np.arange(1, N + 1), N * N),
        'HMACH': HMACH_grid.flatten(),
        'LMACH': LMACH_grid.flatten(),
        'LMACH_MassDep': LMACH_MassDep_grid.flatten()
    })

    # Filter out cells with no halo mass
    src = src[(src['HMACH'] != 0) | (src['LMACH'] != 0) | (src['LMACH_MassDep'] != 0)]

    # Write output file
    output_file = './src_{:.3f}.txt'.format(float(redshift))
    with open(output_file, 'w') as file:
        file.write(f"{len(src)}\n")
        src.to_csv(file, sep=' ', index=False, header=False)
