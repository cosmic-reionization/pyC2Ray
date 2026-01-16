import os
import pickle
import time

import tools21cm as t2c

try:
    import py21cmfast as p21c
except ImportError:
    print(
        "To use the density and halo catalogues from 21cmfast, "
        "install the python version of that code:\n"
        "https://21cmfast.readthedocs.io/en/latest/installation.html"
    )


class Run21cmfast:
    """Modelling the evolution of matter distribution and the dark matter halo."""

    def __init__(
        self,
        HII_DIM,
        BOX_LEN,
        n_jobs=4,
        h=0.67,
        Om=0.31,
        Ob=0.045,
        s8=0.82,
        ns=0.96,
        random_seed=42,
        data_dir="./21cmFAST_data",
        INITIAL_REDSHIFT=300,
        CLUMPING_FACTOR=2.0,
        **kwargs,
    ):
        self.BOX_LEN = BOX_LEN
        self.HII_DIM = HII_DIM
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.INITIAL_REDSHIFT = INITIAL_REDSHIFT
        self.CLUMPING_FACTOR = CLUMPING_FACTOR

        self.set_cosmology(h=h, Om=Om, Ob=Ob, s8=s8, ns=ns)
        self.set_parameters()
        self.create_data_dir(data_dir=data_dir)
        self.IC = None

    def set_cosmology(self, h=0.67, Om=0.31, Ob=0.045, s8=0.82, ns=0.96):
        self.cosmo = {"h": h, "Om": Om, "Ob": Ob, "s8": s8, "ns": ns}

        self.cosmo_params = p21c.CosmoParams(
            SIGMA_8=self.cosmo["s8"],
            hlittle=self.cosmo["h"],
            OMm=self.cosmo["Om"],
            OMb=self.cosmo["Ob"],
            POWER_INDEX=self.cosmo["ns"],
        )

        print("Cosmology for py21cmfast is set to:")
        print(self.cosmo)

    def set_parameters(self):
        params = {
            "HII_DIM": self.HII_DIM,
            "DIM": self.HII_DIM * 3,
            "BOX_LEN": self.BOX_LEN,
            "USE_INTERPOLATION_TABLES": True,
            # "FIXED_IC": True,
            "N_THREADS": self.n_jobs,
        }
        self.user_params = p21c.UserParams(params)

    def create_data_dir(self, data_dir="./21cmFAST_data"):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
            print(f"{data_dir} created.")
        else:
            print(f"{data_dir} already present.")
        self.data_dir = data_dir

    def initialise_21cmfast(self):
        """
        Initialise the 21cmFAST parameters, and check the power spectrum of the initial conditions

        Parameters
        ----------
        param: Bunch
            The parameter file created using the beorn.par().
        data_dir: string
            The dir where to write the 21cmFAST cache data. Default is ./21cmFAST_data.
        Returns
        -------
        IC, pslin, klin : The initial conditions, followed by the power spectrum of the matter field.
        """

        user_params = self.user_params
        cosmo_params = self.cosmo_params

        with p21c.global_params.use(INITIAL_REDSHIFT=300, CLUMPING_FACTOR=2.0):
            IC = p21c.initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed=self.random_seed,
                write=False,  # self.data_dir,
                direc=self.data_dir,
            )

        pslin, klin = t2c.power_spectrum_1d(
            IC.hires_density, kbins=20, box_dims=user_params.BOX_LEN
        )
        self.IC = {"data": IC, "P": pslin, "k": klin}
        return self.IC

    def simulate_matter_21cmfast(self, redshift_list):
        user_params = self.user_params
        cosmo_params = self.cosmo_params
        IC_info = self.IC if self.IC is not None else self.initialise_21cmfast()

        start_time = time.time()
        print("Simulating matter evolution with 21cmFast...")

        dens_dict = {}
        halo_catalog_dict = {}
        with p21c.global_params.use(
            INITIAL_REDSHIFT=self.INITIAL_REDSHIFT, CLUMPING_FACTOR=self.CLUMPING_FACTOR
        ):
            for redshift in redshift_list:
                perturbed_field = p21c.perturb_field(
                    redshift=redshift,
                    init_boxes=IC_info["data"],
                    # user_params=user_params,
                    # cosmo_params=cosmo_params,
                    # astro_params=astro_params,
                    # random_seed=random_seed,
                    write=False,  # self.data_dir,
                    direc=self.data_dir,
                )
                halo_list = p21c.perturb_halo_list(
                    redshift=redshift,
                    init_boxes=IC_info["data"],
                    # user_params=user_params,
                    # cosmo_params=cosmo_params,
                    # astro_params=astro_params,
                    # random_seed=random_seed,
                    write=False,  # self.data_dir,
                    direc=self.data_dir,
                )

                # h0 = cosmo_params.hlittle
                Lbox = user_params.BOX_LEN
                print(
                    "Assuming BOX_LEN is in Mpc. Halo catalogs catalogs have masses in Msol and positions in Mpc."
                )
                dens = perturbed_field.density
                halo_list = {
                    "X": halo_list.halo_coords[:, 0] * Lbox / user_params.HII_DIM,
                    "Y": halo_list.halo_coords[:, 1] * Lbox / user_params.HII_DIM,
                    "Z": halo_list.halo_coords[:, 2] * Lbox / user_params.HII_DIM,
                    "M": halo_list.halo_masses,
                    "z": redshift,
                    "BOX_LEN": Lbox,
                }

                dens_dict[redshift] = dens
                halo_catalog_dict[redshift] = halo_list
                savedata = {
                    "density": dens,
                    "halo_list": halo_list,
                    "cosmo_params": cosmo_params.__dict__,
                    "user_params": user_params.__dict__,
                }
                filename = (
                    self.data_dir
                    + f"/matter_data_{Lbox:.2f}Mpc_{user_params.HII_DIM}_z{redshift:05.2f}.pkl"
                )
                pickle.dump(savedata, open(filename, "wb"))
                print(f"Data saved as {savedata}.")

        end_time = time.time()
        print(f"...done | Runtime = {(end_time - start_time) / 60:.2f} mins")
        return {"dens": dens_dict, "halo_list": halo_catalog_dict}


if __name__ == "__main__":
    HII_DIM = 50
    BOX_LEN = 100
    run_code = Run21cmfast(
        HII_DIM,
        BOX_LEN,
        n_jobs=4,
        h=0.67,
        Om=0.31,
        Ob=0.045,
        s8=0.82,
        ns=0.96,
        random_seed=42,
        data_dir="./21cmFAST_data",
        INITIAL_REDSHIFT=300,
        CLUMPING_FACTOR=2.0,
    )
    redshift_list = [10, 8]
    out_data = run_code.simulate_matter_21cmfast(redshift_list)
