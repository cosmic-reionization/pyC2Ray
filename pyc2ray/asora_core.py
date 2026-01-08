# ===================================================================================================
# This module manages the initialization of the ASORA raytracing extension library. It ensures that
# GPU memory has been allocated when GPU-accelerated functions are called.
# ===================================================================================================

from .load_extensions import load_asora, load_asora_he

libasora = load_asora()
libasora_he = load_asora_he()

__all__ = ["cuda_is_init", "device_init", "device_close", "photo_table_to_device"]

# This flag indicates whether GPU memory has been correctly allocated before calling any methods.
# NOTE: there is no check if the allocated memory has the correct mesh size when calling a function,
# so the user is responsible for that.
cuda_init = False


def cuda_is_init():
    global cuda_init
    return cuda_init


def device_init(N, source_batch_size, rank, nr_gpus):
    """Initialize GPU and allocate memory for grid data

    Parameters
    ----------
    N : int
        Mesh size in grid coordinates
    source_batch_size : int
        Number of sources the GPU handles in parallel. Increasing this parameter
        will speed up raytracing for large numbers of sources, but also increase
        memory usage
    """
    global cuda_init
    if libasora is not None:
        libasora.device_init(N, source_batch_size, rank, nr_gpus)
        cuda_init = True
    else:
        raise RuntimeError("Could not initialize GPU: ASORA library not loaded")


def device_close():
    """Deallocate GPU memory"""
    global cuda_init
    if cuda_init:
        libasora.device_close()
        cuda_init = False
    else:
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )


def photo_table_to_device(thin_table, thick_table):
    """Copy radiation tables to GPU (optically thin & thick tables)"""
    global cuda_init
    if cuda_init:
        libasora.photo_table_to_device(thin_table, thick_table)
    else:
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )
