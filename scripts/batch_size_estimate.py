import argparse
import math
import subprocess
import sys


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse the CLI arguments"""
    parser = argparse.ArgumentParser(
        prog="memory_estimate",
        description="Estimate py2cray run parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-sources",
        type=int,
        default=1_000_000,
        help="number of sources",
    )
    parser.add_argument(
        "-m",
        "--mesh-size",
        type=int,
        default=256,
        help="size of the grid",
    )
    parser.add_argument(
        "-t",
        "--table-size",
        type=int,
        default=2000,
        help="size of each look-up tables",
    )
    parser.add_argument(
        "--with-helium",
        action="store_true",
        help="include He simulation in the estimation",
    )
    parser.add_argument(
        "-f",
        "--num-freq-bins",
        type=int,
        default=100,
        help="size of frequency-dependent histograms",
    )
    parser.add_argument(
        "-R",
        "--radius",
        type=int,
        help="set a maximum radius to reduce memory usage",
    )
    parser.add_argument(
        "--use-single-precision",
        action="store_true",
        help="use single precision floating point",
    )
    parser.add_argument(
        "-i", "--gpu-id", type=int, default=0, help="specify which GPU device"
    )

    return parser.parse_args(args)


def format_bytes(size: int | float) -> str:
    """Format bytes into human readable format"""
    for unit in ("", "Ki", "Mi", "Gi"):
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:3.2f} {unit}B"


def get_memory_available(gpu_id: int = 0) -> int:
    """Return the memory availble to the user in MiB"""
    ret = subprocess.check_output(
        (
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.reserved",
            "--format=csv,noheader,nounits",
        )
    )

    mem_total, _, mem_reserve = ret.decode("utf-8").partition(",")
    return (int(mem_total) - int(mem_reserve)) * 1024**2


def get_octahedron_size(R: int, mesh_size: int) -> int:
    q_max = math.ceil(min(math.sqrt(3) * R, 1.5 * mesh_size))
    q0 = 1 + 2 * q_max
    return q0 * (2 * q_max**2 + q0 + 2) // 3


def print_mem_reqs(
    memory: int,
    mesh_size: int,
    num_sources: int,
    table_size: int,
    helium: bool = False,
    num_freq_bins: int = 100,
    radius: None | int = None,
    prec: int = 8,
) -> None:
    """Print memory requirements for the input parameters"""
    ns_mem = (12 + prec) * num_sources
    pt_mem = 2 * prec * table_size
    if helium:
        pt_mem *= 2
    grid_mem = prec * mesh_size**3

    xs_mem = grid_mem
    if helium:
        xs_mem *= 3

    phi_mem = grid_mem
    if helium:
        phi_mem *= 6

    print(
        f"""
Required parameters:
 - mesh size: {mesh_size}
 - number of sources: {num_sources:,}
 - look-up table size: {table_size}
 - with helium: {"yes" if helium else "no"}"""
    )
    if helium:
        print(f" - frequency bins: {num_freq_bins}")

    print(
        f"""
GPU {args.gpu_id} has total memory of {format_bytes(memory_total)}

> Memory required for sources: {format_bytes(ns_mem)}
> Memory required for LUTs: {format_bytes(pt_mem)}
> Memory required for density mesh: {format_bytes(grid_mem)}
> Memory required for fraction meshes: {format_bytes(xs_mem)}
> Memory required for photoionization grids: {format_bytes(xs_mem)}"""
    )
    base_mem = ns_mem + pt_mem + grid_mem + xs_mem + phi_mem
    if radius is None:
        source_mem = grid_mem
    else:
        source_mem = prec * get_octahedron_size(radius, mesh_size)

    if helium:
        xsec_mem = 3 * prec * num_freq_bins
        base_mem += xsec_mem
        source_mem *= 3
        print(
            f"> Memory required for frequency-dependent histograms: {format_bytes(xsec_mem)}"
        )

    print(
        f"""
> Baseline memory required: {format_bytes(base_mem)}
> Buffer memory required per source: {format_bytes(source_mem)}"""
    )

    free_mem = memory_total - base_mem
    if free_mem < 0:
        print("\nNot enough memory for these parameters.")
        return

    batches = free_mem / source_mem
    n_batches = int(math.ceil(num_sources / batches))
    print(
        f"""
Available memory {format_bytes(free_mem)} can fit a batch size of {batches:.0f}
Number of batch submissions: {n_batches:,}
"""
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    memory_total = get_memory_available(args.gpu_id)

    prec = 4 if args.use_single_precision else 8
    print_mem_reqs(
        memory_total,
        args.mesh_size,
        args.num_sources,
        args.table_size,
        args.with_helium,
        args.num_freq_bins,
        args.radius,
        prec,
    )
