from contextlib import contextmanager

import astropy.constants as cst
import astropy.units as u
import numpy as np
import pytest

from pyc2ray.load_extensions import load_asora
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table

asora = load_asora()

if asora is None:
    pytest.skip("libasora.so missing, skipping tests", allow_module_level=True)


@contextmanager
def device_init(mesh_size: int = 50, batch_size: int = 8):
    asora.device_init(mesh_size, batch_size, 0, 1)
    try:
        yield
    finally:
        asora.device_close()


def test_device_init():
    with device_init():
        pass


@contextmanager
def setup_do_all_sources(
    mesh_size: int = 50, batch_size: int = 8, num_sources: int = 10
):
    with device_init(mesh_size, batch_size):
        R_max = 15.0

        # Calculate the table
        minlog_tau, maxlog_tau, num_tau = -20.0, 4.0, 20000
        tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)

        # HI cross section at its ionzing frequency (weighted by freq_factor)
        sigma_HI_at_ion_freq = np.float64(6.30e-18)

        # Min and max frequency of the integral
        freq_min, freq_max = (
            (13.598 * u.eV / cst.h).to("Hz").value,
            (54.416 * u.eV / cst.h).to("Hz").value,
        )
        radsource = BlackBodySource(1e5, False, freq_min, sigma_HI_at_ion_freq)
        photo_thin_table, photo_thick_table = radsource.make_photo_table(
            tau, freq_min, freq_max, 1e48
        )

        # Allocate tables to GPU device
        asora.photo_table_to_device(photo_thin_table, photo_thick_table, num_tau)

        size = mesh_size**3
        coldensh_out = np.zeros(size, dtype=np.float64)
        phi_ion = np.zeros(size, dtype=np.float64)
        ndens = np.full(size, 1e-3, dtype=np.float64)
        xHII = np.full(size, 1e-4, dtype=np.float64)

        # Copy density field to GPU device
        asora.density_to_device(ndens, mesh_size)

        # Efficiency factor (converting mass to photons)
        f_gamma = 100.0

        # Define some random sources
        rng = np.random.default_rng(918)
        src_pos = rng.integers(0, mesh_size, size=(3 * num_sources), dtype=np.int32)
        norm_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
        norm_flux *= f_gamma / 1e48

        # Copy source list to GPU device
        asora.source_data_to_device(src_pos, norm_flux, num_sources)

        # Size of a cell
        box = 50.0 * u.pc
        dr = (box / mesh_size).cgs.value

        yield (
            R_max,
            coldensh_out,
            sigma_HI_at_ion_freq,
            dr,
            ndens,
            xHII,
            phi_ion,
            num_sources,
            mesh_size,
            minlog_tau,
            dlogtau,
            num_tau,
        )


def test_do_all_sources():
    with setup_do_all_sources() as args:
        asora.do_all_sources(*args)

        expected_phi_ion = np.load("tests/data/photo_ionization_rate.npy")

        phi_ion = args[6] * 1e40
        expected_phi_ion *= 1e40

        assert np.allclose(phi_ion, expected_phi_ion)


@pytest.mark.parametrize("mesh_size", [32, 64, 128])
@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("num_sources", [100, 1000, 10000])
def test_benchmark_do_all_sources(benchmark, mesh_size, batch_size, num_sources):
    # Manual warmup
    with setup_do_all_sources(mesh_size, batch_size, num_sources) as args:
        asora.do_all_sources(*args)
    # Run the actual benchmark
    with setup_do_all_sources(mesh_size, batch_size, num_sources) as args:
        benchmark(asora.do_all_sources, *args)
