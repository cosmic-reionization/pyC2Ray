import numpy as np
import glob, os

def get_redshifts_from_output(output_dir, z_low=None, z_high=None, bracket=False):
    output_files = glob.glob(os.path.join(output_dir,'xfrac*'))

    redshifts = []
    for f in output_files:
        try:
            z = float(f.split('_')[-1][:-4])
            redshifts.append(z)
        except: 
            pass
    
    return np.sort(np.array(redshifts))

def find_bins(input_array, binning_array):
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


def get_source_redshifts(source_dir, z_low = None, z_high = None, bracket=False):
        # TODO: this is temporary, waiting new release of tools21cm. Once done we can simply use t2c.get_source_redshifts

        ''' 
        Make a list of the redshifts of all the xfrac files in a directory.
        
        Parameters:
                * xfrac_dir (string): the directory to look in
                * z_low = None (float): the minimum redshift to include (if given)
                * z_high = None (float): the maximum redshift to include (if given)
                * bracket = False (bool): if true, also include the redshifts on the
                        lower side of z_low and the higher side of z_high
         
        Returns: 
                The redhifts of the files (numpy array of floats) '''

        source_files = glob.glob(os.path.join(source_dir,'*-coarsest_wsubgrid_sources.dat'))

        redshifts = []
        for f in source_files:
                try:
                        z = float(f[f.rfind('/')+1:f.rfind('-coarsest_wsubgrid_sources')])
                        redshifts.append(z)
                except:
                        pass

        return _get_redshifts_in_range(redshifts, z_low, z_high, bracket)

def get_same_values_in_array(arr1, arr2):
    """ return values in common between two arrays in decreasing order"""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    interp_arr = np.sort(np.array(list(set(arr1).intersection(arr2))))[::-1]
    return interp_arr


def _get_redshifts_in_range(redshifts, z_low, z_high, bracket):
        '''
        Filter out redshifts outside of range. For internal use.
        '''
        redshifts = np.array(redshifts)
        redshifts.sort()
        if bracket:
                if z_low < redshifts.min() or z_high > redshifts.max():
                        raise Exception('No redshifts to bracket range.')
                z_low = redshifts[redshifts <= z_low][-1]
                z_high = redshifts[redshifts >= z_high][0]
        if z_low == None:
                z_low = redshifts.min()-1
        if z_high == None:
                z_high = redshifts.max()+1
        idx = (redshifts >= z_low)*(redshifts <= z_high)
        redshifts = redshifts[idx]

        return np.array(redshifts)