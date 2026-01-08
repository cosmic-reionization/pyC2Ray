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


@pytest.fixture
def init_device():
    asora.device_init()
    yield
    asora.device_close()


def test_device_init(init_device):
    asora.is_device_init()


@contextmanager
def setup_do_all_sources(
    num_sources: int = 10, mesh_size: int = 50, batch_size: int = 8, block_size=256
):
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
    asora.photo_table_to_device(photo_thin_table, photo_thick_table)

    size = mesh_size**3
    phi_ion = np.empty(size, dtype=np.float64)
    ndens = np.full(size, 1e-3, dtype=np.float64)
    xHII = np.full(size, 1e-4, dtype=np.float64)

    # Copy density field to GPU device
    asora.density_to_device(ndens)

    # Efficiency factor (converting mass to photons)
    f_gamma = 100.0

    # Define some random sources
    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, mesh_size, size=(3 * num_sources), dtype=np.int32)
    norm_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
    norm_flux *= f_gamma / 1e48

    # Copy source list to GPU device
    asora.source_data_to_device(src_pos, norm_flux)

    # Size of a cell
    box = 50.0 * u.pc
    dr = (box / mesh_size).cgs.value

    yield (
        R_max,
        sigma_HI_at_ion_freq,
        dr,
        xHII,
        phi_ion,
        num_sources,
        mesh_size,
        minlog_tau,
        dlogtau,
        num_tau,
        batch_size,
        block_size,
    )


def test_do_all_sources(data_dir, init_device):
    with setup_do_all_sources() as args:
        asora.do_all_sources(*args)

        expected_phi_ion = np.load(data_dir / "photo_ionization_rate.npy")

        phi_ion = args[4] * 1e40
        expected_phi_ion *= 1e40

        assert np.allclose(phi_ion, expected_phi_ion)


@pytest.mark.parametrize("mesh_size", [32, 64, 128])
@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("block_size", [128, 256, 512])
@pytest.mark.benchmark(warmup=True, warmup_iterations=1)
def test_benchmark_do_all_sources(
    benchmark, init_device, mesh_size, batch_size, block_size
):
    with setup_do_all_sources(10000, mesh_size, batch_size, block_size) as args:
        benchmark(asora.do_all_sources, *args)
