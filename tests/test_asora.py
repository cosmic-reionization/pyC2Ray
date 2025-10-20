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


def test_device_init():
    asora.device_init(100, 8, 0, 1)
    asora.device_close()


def test_do_all_sources(request):
    N = 50
    # allocate GPU memory for the grid and sources batch size
    asora.device_init(N, 8, 0, 1)

    # calculate the table
    num_tau = 20000
    minlog_tau, maxlog_tau = -20.0, 4.0
    tau, dlogtau = make_tau_table(minlog_tau, maxlog_tau, num_tau)

    # HI cross section at its ionzing frequency (weighted by freq_factor)
    sigma_HI_at_ion_freq = np.float64(6.30e-18)

    # min and max frequency of the integral
    freq_min, freq_max = (
        (13.598 * u.eV / cst.h).to("Hz").value,
        (54.416 * u.eV / cst.h).to("Hz").value,
    )

    radsource = BlackBodySource(1e5, False, freq_min, sigma_HI_at_ion_freq)
    photo_thin_table, photo_thick_table = radsource.make_photo_table(
        tau, freq_min, freq_max, 1e48
    )

    # allocate tables to GPU device
    asora.photo_table_to_device(photo_thin_table, photo_thick_table, num_tau)

    coldensh_out = np.zeros((N, N, N), dtype="float64").ravel()
    phi_ion = np.zeros((N, N, N), dtype="float64").ravel()  # in ^-1
    ndens = np.full((N, N, N), 1e-3, dtype=np.float64).ravel()  # g/cm^3
    xHII = np.full((N, N, N), 1e-4, dtype=np.float64).ravel()

    # copy density field to GPU device
    asora.density_to_device(ndens, N)

    # efficiency factor (converting mass to photons)
    f_gamma = 100.0

    # define some random sources
    num_src = 10
    rng = np.random.default_rng(918)
    src_pos = rng.integers(0, N, size=(3, num_src), dtype=np.int32).ravel()
    norm_flux = rng.uniform(1e10, 1e14, size=num_src).astype(np.float64)
    norm_flux *= f_gamma / 1e48

    # copy source list to GPU device
    asora.source_data_to_device(src_pos, norm_flux, num_src)

    boxsize = 50.0 * u.pc
    dr = (boxsize / N).cgs.value

    asora.do_all_sources(
        15.0,
        coldensh_out,
        sigma_HI_at_ion_freq,
        dr,
        ndens,
        xHII,
        phi_ion,
        num_src,
        N,
        minlog_tau,
        dlogtau,
        num_tau,
    )

    expected_phi_ion = np.load("tests/data/photo_ionization_rate.npy")

    phi_ion *= 1e40
    expected_phi_ion *= 1e40

    assert np.allclose(phi_ion, expected_phi_ion)

    asora.device_close()
