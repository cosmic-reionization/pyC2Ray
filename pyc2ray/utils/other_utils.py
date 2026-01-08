import glob
import os
import time

import numpy as np
from scipy import stats


def get_extension_in_folder(path):
    arr = glob.glob(path + "xfrac*")
    f = arr[0]
    if os.path.isfile(f):
        ext = f[f.rfind(".") :]
    elif os.path.isdir(f):
        f = arr[1]
        ext = f[f.rfind(".") :]
    else:
        ValueError("Check the output directory, it maybe empty or do not exist")
    return ext


def get_redshifts_from_output(output_dir, z_low=None, z_high=None, bracket=False):
    """from a given directory get the redshift based on the name of the file (remark: file name must be in the form of 'xfrac_%.3f.[extension]')"""
    output_files = glob.glob(os.path.join(output_dir, "xfrac*"))

    redshifts = []
    for f in output_files:
        try:
            z = float(f.split("z")[-1][:-4])
            redshifts.append(z)
        except Exception:
            pass

    return np.sort(np.array(redshifts))[::-1]


def find_bins(input_array, binning_array):
    """For a given bin array and an input array get the indexes of the bins for each element of the input array"""
    # Sort the binning array in ascending order
    sorted_bins = np.sort(binning_array)

    left_bins = []
    right_bins = []

    if isinstance(input_array, (np.ndarray, list)):
        for value in input_array:
            # Find the index where the value should be inserted in the sorted_bins
            bin_index = np.digitize(value, sorted_bins)

            # Check if bin_index is within the bounds of the sorted_bins
            if bin_index > 0 and bin_index <= len(sorted_bins):
                left_bin = sorted_bins[bin_index - 1]
                right_bin = sorted_bins[bin_index]
            elif bin_index == 0:
                left_bin = None
                right_bin = sorted_bins[bin_index]
            else:
                left_bin = sorted_bins[bin_index - 1]
                right_bin = None

            left_bins.append(left_bin)
            right_bins.append(right_bin)
        return np.array(left_bins), np.array(right_bins)

    else:
        value = input_array
        # Find the index where the value should be inserted in the sorted_bins
        bin_index = np.digitize(value, sorted_bins)

        # Check if bin_index is within the bounds of the sorted_bins
        if bin_index > 0 and bin_index <= len(sorted_bins):
            left_bin = sorted_bins[bin_index - 1]
            right_bin = sorted_bins[bin_index]
        elif bin_index == 0:
            left_bin = None
            right_bin = sorted_bins[bin_index]
        else:
            left_bin = sorted_bins[bin_index - 1]
            right_bin = None

        left_bins.append(left_bin)
        right_bins.append(right_bin)

        return left_bins[0], right_bins[0]


def get_source_redshifts(source_dir, z_low=None, z_high=None, bracket=False):
    # TODO: this is temporary, waiting new release of tools21cm. Once done we can simply use t2c.get_source_redshifts

    """
    Make a list of the redshifts of all the xfrac files in a directory.

    Parameters:
            * xfrac_dir (string): the directory to look in
            * z_low = None (float): the minimum redshift to include (if given)
            * z_high = None (float): the maximum redshift to include (if given)
            * bracket = False (bool): if true, also include the redshifts on the
                    lower side of z_low and the higher side of z_high

    Returns:
            The redhifts of the files (numpy array of floats)"""

    source_files = glob.glob(
        os.path.join(source_dir, "*-coarsest_wsubgrid_sources.dat")
    )

    redshifts = []
    for f in source_files:
        try:
            z = float(f[f.rfind("/") + 1 : f.rfind("-coarsest_wsubgrid_sources")])
            redshifts.append(z)
        except Exception:
            pass

    return _get_redshifts_in_range(redshifts, z_low, z_high, bracket)


def get_same_values_in_array(arr1, arr2):
    """Return values in common between two arrays in decreasing order"""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    interp_arr = np.sort(np.array(list(set(arr1).intersection(arr2))))[::-1]
    return interp_arr


def _get_redshifts_in_range(redshifts, z_low, z_high, bracket):
    """Filter out redshifts outside of range. For internal use."""
    redshifts = np.array(redshifts)
    redshifts.sort()
    if bracket:
        if z_low < redshifts.min() or z_high > redshifts.max():
            raise Exception("No redshifts to bracket range.")
        z_low = redshifts[redshifts <= z_low][-1]
        z_high = redshifts[redshifts >= z_high][0]
    if z_low is None:
        z_low = redshifts.min() - 1
    if z_high is None:
        z_high = redshifts.max() + 1
    idx = (redshifts >= z_low) * (redshifts <= z_high)
    redshifts = redshifts[idx]

    return np.array(redshifts)


def bin_sources(srcpos_mpc, mstar_msun, boxsize, meshsize):
    # define bin for position of sources
    mesh_bin = np.linspace(0, boxsize, meshsize + 1)

    # sum toghete the mass of sources
    binned_mass, _, _ = stats.binned_statistic_dd(
        srcpos_mpc, mstar_msun, statistic="sum", bins=[mesh_bin, mesh_bin, mesh_bin]
    )

    # get a list of the source positon and mass
    srcpos = np.argwhere(binned_mass > 0)
    srcmstar = binned_mass[binned_mass > 0]

    return srcpos, srcmstar


def display_time(time_in_seconds):
    """Return a string that display nicely the lapsed time"""
    hrs, residual = divmod(time_in_seconds, 3600.0)
    mins, secs = divmod(residual, 60.0)
    if hrs == 0:
        if mins == 0:
            display = "%.2fs" % (secs)
        else:
            display = "%dm %.2fs" % (mins, secs)
    else:
        display = "%dh %dm %.2fs" % (hrs, mins, secs)
    return display


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None
        self._prevlap_time = None
        self._summary = "\n--- TIMER SUMMARY ---\n"
        self._lap_counter = 0

    def _display(self, time_in_seconds):
        hrs, residual = divmod(time_in_seconds, 3600.0)
        mins, secs = divmod(residual, 60.0)
        if hrs == 0:
            if mins == 0:
                display = "%.2fs" % (secs)
            else:
                display = "%dm %.2fs" % (mins, secs)
        else:
            display = "%dh %dm %.2fs" % (hrs, mins, secs)
        return display

    def start(self):
        """Start the timer"""
        # check if there isn't another timer running
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        # register starting time
        self._start_time = time.perf_counter()

    def lap(self, mess=None):
        """Register a lap time"""
        # increase the laps counter
        self._lap_counter += 1

        # check if the timer is initialized
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # register the lap time
        lap_time = time.perf_counter()
        if self._prevlap_time is not None:
            elapsed_time = lap_time - self._prevlap_time
        else:
            elapsed_time = lap_time - self._start_time

        # register the time since the start
        elapsed_time_from_start = lap_time - self._start_time

        # overwrite previous lap time for next one
        self._prevlap_time = lap_time

        # register a message
        mess = "- " + str(mess) if mess is not None else ""
        self._summary += " step %d: %s %s\n" % (
            self._lap_counter,
            self._display(elapsed_time),
            mess,
        )

        # return elapsed time since the start
        return self._display(elapsed_time_from_start)

    def stop(self, mess=""):
        """Stop the timer, and report the elapsed time"""
        # check if the timer is initialized
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # register the end time
        time_stop = time.perf_counter()

        # calculate the elapsed time since the start
        elapsed_time = time_stop - self._start_time

        # write the timer summary message
        mess = " - " + str(mess) if mess != "" else mess
        self.summary = self._summary + "Elapsed time: %s %s" % (
            self._display(elapsed_time),
            mess,
        )

        # init the start time in case you want to start a new timer in the same run
        self._start_time = None
