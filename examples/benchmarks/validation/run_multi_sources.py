"""Multi-source cosmological test, same loop as pyc2ray_gpu_validation_dardel.ipynb
(section 3), without plotting. Usage, from examples/benchmarks (results_basename
in the parameter file is relative):
  python validation/run_multi_sources.py <paramfile> <srcfile>"""
import sys

import astropy.units as u
import pyc2ray as pc2r

num_steps_between_slices = 2
numzred = 10

sim = pc2r.C2Ray_Test(paramfile=sys.argv[1])
zred_array = sim.generate_redshift_array(numzred, 1e7)
srcfile = sys.argv[2]
srcpos, srcflux = sim.read_sources(srcfile, 5)

timer = pc2r.Timer()
timer.start()
for k in range(len(zred_array) - 1):
    zi, zf = zred_array[k], zred_array[k + 1]
    dt = sim.set_timestep(zi, zf, num_steps_between_slices)
    sim.write_output(zi)
    sim.set_constant_average_density(ndens=sim.avg_dens, z=zi)
    for t in range(num_steps_between_slices):
        if sim.cosmological:
            sim.cosmo_evolve(dt)
        sim.evolve3D(dt, srcflux, srcpos)
        timer.lap("t=%d" % t)
sim.write_output(zf)
timer.stop()
print(timer.summary)
