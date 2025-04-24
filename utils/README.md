The following folder contains some basic utils to generate the density fields required as inpots for the pyc2ray_ramses class.

If postprocessing **RAMSES** simulations:
* Run the part2cube.f90 routine on your snapshots (this is provided in the ramses code ./utils/f90/)
* Run the produce_baryonic_dens_RAMSES.py on the generated cube files

If postprocessing **AREPO** DMO sims:
* Run cic.exe found in CIC-UniformDMO (if running on a uniform DMO sim) or CIC-ZoomSims (if running on zoom sims)
* Run the produce_baryonic_dens_arepo.py on the generated \*.BCIC.\* files
