import os
from glob import glob

import h5py
import numpy as np
import tools21cm as t2c

# ==================================================
# Script to Batch-convert CubeP3M raw binary source
# files to HDF5 format
# ==================================================

sort = False
path_in = "/store/ska/sk015/244Mpc_RT/244Mpc_f2_0_250/sources/"
path_out = "/scratch/snx3000/mibianco/results_c2ray/convert_sources/"

os.chdir(path_in)
files = glob("*.dat")

for fname in files:
    # We assume that the file extension is .dat
    fname_noext = fname[:-4]
    fname_hdf5 = path_out + fname_noext + ".hdf5"

    if not (os.path.exists(fname_hdf5)):
        print("Converting " + fname + "...")
        srcfile_raw = t2c.SourceFile(fname)

        z = srcfile_raw.z
        masstype = srcfile_raw.mass
        numsrc = srcfile_raw.sources_list.shape[0]

        # if args.sort:
        if sort:
            # We sort the sources from strongest to weakest
            idx_sorted = np.flip(np.argsort(srcfile_raw.sources_list[:, 3]))
            sources_pos = srcfile_raw.sources_list[idx_sorted, 0:3]
            sources_mass = srcfile_raw.sources_list[idx_sorted, 3]
        else:
            sources_pos = srcfile_raw.sources_list[:, 0:3]
            sources_mass = srcfile_raw.sources_list[:, 3]

        non_zero = sources_mass != 0

        # Create HDF5 file from the data
        with h5py.File(fname_hdf5, "w") as f:
            # Store Data
            dset_pos = f.create_dataset("sources_positions", data=sources_pos[non_zero])
            dset_mass = f.create_dataset("sources_mass", data=sources_mass[non_zero])

            # Store Metadata
            f.attrs["z"] = z
            f.attrs["masstype"] = masstype
            f.attrs["filename"] = fname_hdf5
            dset_mass.attrs["unit"] = "Solar Mass"
    else:
        print("File " + fname_hdf5 + " exist... skip")

print("done.")
