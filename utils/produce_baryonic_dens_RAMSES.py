# Program whihh provided a cube_*.dat file from a DMO sim (in RAMSES) converts it to a 
# baryonic denisty field
# The cube_*.dat file can be created using the part2cube routine in the ramses utils/f90 files.

import numpy as np
import glob 
import re

path_to_cube = './'
path_to_snap = '/p/scratch/lgreion/david/runs/zoom_pyc2ray/coarser_densities/dens/'
save_path = '/p/scratch/lgreion/david/runs/zoom_pyc2ray/coarser_densities/'
start_snap = 20
end_snap = 49

Ngrid = 256 # size of the mesh

# Cosmology
h = 0.677
OmegaB = 0.048
OmegaM = 0.318 
box_size = 100 #Mpc/h


for i in range(start_snap,end_snap+1): 
    
    with open(f'./dens/cube_counts_{i:05d}.dat', 'r') as f:
        trail_byte = np.fromfile(f, dtype=np.int32, count=1)  # Should be 12 for the 3 integers (4 bytes each)
        
        print('trail_byte', trail_byte)

        # Read the dimensions nx, ny, nz (assuming they are written as integers)
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)

        trail_byte = np.fromfile(f, dtype=np.int32, count=2)  # Should be 12 for the 3 integers (4 bytes each)
        
        print('trail_byte', trail_byte)
        
        # Allocate the toto array (real(KIND=8) corresponds to float64 in Python)
        toto = np.fromfile(f, dtype=np.float64, count=nx*ny*nz)
        
        # Reshape the flat array into the 3D array (Fortran writes arrays column-major)
        cube = toto.reshape((nx, ny, nz), order='F')  # Use 'F' for Fortran-style ordering
        
        pc=  3.086e18 #1 pc in cm
        Mpc = 1e6*pc
        G_grav = 6.6732e-8

        H0 = 100.0*h
        H0cgs = H0*1e5/Mpc
        
        cube = cube/np.sum(cube)

        rho_crit_0 = 3.0*H0cgs*H0cgs/(8.0*np.pi*G_grav)

        # Matter density field in (comoving) g/cm^3
        baryonic_dens = cube*rho_crit_0*OmegaB*(Ngrid)**3 

    # Pattern to match files ending with "_halos"
    AHF_directory = '/p/scratch/lgreion/david/analysis_hestiaRT/DMO_256_4096/AHF/{0:03d}/halos'.format(i)
    pattern = AHF_directory + "/*_halos"

    # Get list of files ending with "_halos"
    halos_files = glob.glob(pattern)
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

    np.save(save_path+'dens_cgs_{:.3f}.npy'.format(z), baryonic_dens)
    print(z)


print('Theoretical value: ',rho_crit_0*OmegaB )
print('Avg value in sim: ', np.average(baryonic_dens))

    
