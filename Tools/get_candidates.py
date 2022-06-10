import Tools.data_movement as dm
import Tools.data_processing as dp
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import math
import pandas as pd
import copy
from Tools import auxiliary_functions

"""
This file provides a bunch of wrappers for scipy find_peaks for specific peak types

Mainly used in data augmentation
These were taken from the initial detect peak classifiers
"""

# Helpers
def low_pass_filter(data, window_len):
    """
    (2 * window_len) + 1 is the size of the window that determines the values that
    influence the current measurement (middle of window)
    """
    kernel = np.lib.pad(np.linspace(1, 3, window_len), (0, window_len - 1), "reflect")
    kernel = np.divide(kernel, np.sum(kernel))
    return ndimage.convolve(data, kernel)


def isInRange(indx, remove_ranges):
    for rng in remove_ranges:
        if rng[0] <= indx and indx <= rng[1]:
            return True
    return False


def get_candidates(data: np.ndarray, params: dict):

    """
    Return all peaks that should be scanned for out or order peaks(oop)
    We don't want to return skyrocketing peaks/local fluctuations - although they are oop,
    they will be caught by their respective algorithms

    data   : timeseries to scan for peaks
    return : peaks indentified with given hyperparameters and properties of those peaks
    """
    peaks, props = find_peaks(
        data[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=params["dist"],
        prominence=params["prom"],
        width=params["width"],
        wlen=params["wlen"],
        rel_height=params["rel_h"],
    )
    return peaks, props


########### FDOM ##########
def get_cands_fDOM_PP(fdom_filename, truths_filename, is_augmented=False):
    """
    Get candidates from fDOM phantom peaks

    PARAMS:
    fdom_filename: fdom file name
    truths: truths fdom file name
    is_augmented: if data is augmented
    """

    # pass fDOM data through low pass filter
    if not is_augmented:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))
    smoothed_signal = low_pass_filter(fDOM_data[:, 1], 7)
    fDOM_data = np.column_stack((fDOM_data[:, 0], smoothed_signal))

    candidate_params = {
        "prom": [3, None],
        "width": [None, None],
        "wlen": 200,
        "dist": 1,
        "rel_h": 0.6,
    }

    # if data is augmented, no remove ranges
    if not is_augmented:
        remove_ranges = [[17816, 17849], [108170, 108200], [111364, 111381]]
    else:
        remove_ranges = [[-1, -1]]

    peaks, props = get_candidates(fDOM_data, candidate_params)

    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    # Remove erroneously detected peaks
    temp = []
    for peak in cands:
        if not (isInRange(peak[0], remove_ranges)):
            temp.append(peak)
    cands = copy.deepcopy(temp)

    cands_df = pd.DataFrame(cands)

    # now load in ground truths, and drop all things in cands that are not anomaly peaks
    truths = pd.read_csv(truths_filename)

    # drop all NPP indices
    truths = truths[truths["label_of_peak"] == "PP"]

    # drop all rows in cands that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # reindex frame
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    # return frame
    return cands_df


def get_cands_fDOM_SKP(fdom_filename, truths_filename, is_augmented=False):
    """
    Get candidates from fDOM Skyrocketing peaks

    PARAMS:
    fdom_filename: fdom file name
    truths: truths fdom file name
    """

    # load in fdom data
    if not is_augmented:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))

    # find peaks
    prominence_range = [5, None]
    width_range = [None, None]
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        fDOM_data[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_df = pd.DataFrame(cands)

    # import truths
    truths = pd.read_csv(truths_filename)

    # drop all NSKP indices
    truths = truths[truths["label_of_peak"] == "SKP"]

    # drop all rows in cands that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df


def get_cands_fDOM_PLP(fdom_filename, truths_filename, is_augmented=False):
    """
    Get candidates from fDOM plummeting peaks

    PARAMS:
    fdom_filename: fdom file name
    truths: truths fdom file name
    """

    # load data
    if not is_augmented:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))

    # flip fdom
    flipped_fDOM = dp.flip_timeseries(copy.deepcopy(fDOM_data))

    # Get fDOM plummeting peak candidate set using scipy find_peaks()
    prominence_range = [3, None]  # peaks must have at least prominence 3
    width_range = [None, 10]  # peaks cannot have a base width of more than 5
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        flipped_fDOM[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i] * -1,  # multiply by -1 as peak is downward
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_df = pd.DataFrame(cands)

    # get truths
    truths = pd.read_csv(truths_filename)

    # drop all non PLP indices
    truths = truths[truths["label_of_peak"] == "PLP"]

    # drop all rows in cands not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    # return data
    return cands_df


def get_cands_fDOM_FPT(
    fdom_filename, truths_filename, is_augmented=False, lookup_csv_filename=None
):
    """
    get candidates for fDOM flat plateaus

    NOTE: these values are HARD CODED CURRENTLY, as I do not have a good way to detect plateaus
    NOTE: idx of peak is just the left base, as it is a flat plateau
    """

    if not is_augmented:
        # load raw original data with hardcoded vals
        cands = [[219005, 219005, 219578, 24.5342], [212951, 212951, 213211, 26.54534]]

        cands_df = pd.DataFrame(cands)
        cands_df.columns = ["idx_of_peak", "left_base", "right_base", "amplitude"]

    else:
        # use augmented vals
        cands_df = pd.read_csv(lookup_csv_filename)

    return cands_df


def get_cands_fDOM_FSK(
    fdom_filename, truths_filename, is_augmented=False, lookup_csv_filename=None
):
    """
    get candidates for fDOM flat sinks

    NOTE: these values are HARD CODED CURRENTLY, as I do not have a good way to detect plateaus
    NOTE: idx of peak is just the left base, as it is a flat sink
    """

    if not is_augmented:
        cands = [[85747, 85747, 86462, 10.54653]]

        cands_df = pd.DataFrame(cands)
        cands_df.columns = ["idx_of_peak", "left_base", "right_base", "amplitude"]

    else:
        cands_df = pd.read_csv(lookup_csv_filename)

    return cands_df


def get_cands_fDOM_NAP(
    fdom_filename,
    truths_filename,
    augmented_data=False,
    fpt_lookup_filename=None,
    fsk_lookup_filename=None,
):
    """
    get candidates from non anomaly peak data in fdom

    PARAMS:
    fdom_filename: fdom file name
    truths: truths fdom file name
    augmented_data: whether the file passed in is augmented or not
    """

    # get all peaks from other data types, drop dupes, go from there
    # get peaks from NPP:
    # pass fDOM data through low pass filter
    if not augmented_data:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))

    smoothed_signal = low_pass_filter(fDOM_data[:, 1], 7)
    fDOM_data = np.column_stack((fDOM_data[:, 0], smoothed_signal))

    candidate_params = {
        "prom": [3, None],
        "width": [None, None],
        "wlen": 200,
        "dist": 1,
        "rel_h": 0.6,
    }

    remove_ranges = [[17816, 17849], [108170, 108200], [111364, 111381]]

    peaks, props = get_candidates(fDOM_data, candidate_params)

    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    # Remove erroneously detected peaks
    temp = []
    for peak in cands:
        if not (isInRange(peak[0], remove_ranges)):
            temp.append(peak)
    cands = copy.deepcopy(temp)

    cands_npp = pd.DataFrame(cands)

    # get peaks from NPLP:
    # load data
    if not augmented_data:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))

    # flip fdom
    flipped_fDOM = dp.flip_timeseries(copy.deepcopy(fDOM_data))

    # find peaks

    # Get fDOM plummeting peak candidate set using scipy find_peaks()
    prominence_range = [3, None]  # peaks must have at least prominence 3
    width_range = [None, 10]  # peaks cannot have a base width of more than 5
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        flipped_fDOM[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_nplp = pd.DataFrame(cands)

    # get peaks from NSKP:
    # load in fdom data
    if not augmented_data:
        fDOM_data = dm.read_in_preprocessed_timeseries(fdom_filename)
    else:
        fDOM_data = np.array(dm.read_in_timeseries(fdom_filename, True))

    # find peaks
    prominence_range = [5, None]
    width_range = [None, None]
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        fDOM_data[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_nskp = pd.DataFrame(cands)

    # get cands from NFPT
    cands_nfpt = get_cands_fDOM_FPT(
        fdom_filename, truths_filename, augmented_data, fpt_lookup_filename
    )

    cands_nfpt = cands_nfpt.rename(
        columns={"idx_of_peak": 0, "left_base": 1, "right_base": 2, "amplitude": 3}
    )

    # get cands from NFSK
    cands_nfsk = get_cands_fDOM_FSK(
        fdom_filename, truths_filename, augmented_data, fsk_lookup_filename
    )

    cands_nfsk = cands_nfsk.rename(
        columns={"idx_of_peak": 0, "left_base": 1, "right_base": 2, "amplitude": 3}
    )

    cands_df = pd.concat([cands_nskp, cands_npp, cands_nplp, cands_nfpt, cands_nfsk])

    cands_df = cands_df.sort_values(by=[0], kind="stable")

    # cands_df = cands_df[~cands_df.index.duplicated(keep="first")]

    cands_df = cands_df.reset_index(drop=True)

    # import ground truths
    truths = pd.read_csv(truths_filename)

    truths = truths[truths["label_of_peak"] == "NAP"]

    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df


######## TURBIDITY #########
def get_cands_turb_PP(
    turb_filename, truths_filename, is_augmented=False, delete_dat_range=None
):
    """
    Get candidates from turbidity phantom peaks
    """

    # load data
    if not is_augmented:
        turb_data = dm.read_in_preprocessed_timeseries(turb_filename)
    else:
        turb_data = np.array(dm.read_in_timeseries(turb_filename, True))

    turb_cand_params = {
        "prom": [6, None],
        "width": [None, None],
        "wlen": 200,
        "dist": 1,
        "rel_h": 0.6,
    }

    turb_peaks, turb_props = get_candidates(turb_data, turb_cand_params)

    if delete_dat_range == None:
        delete_dat_range = "Data/misc/flat_plat_ranges.txt"

    # if not augmented data, we need to delete the missing range
    if not is_augmented:
        turb_peaks, turb_props = dp.delete_missing_data_peaks(
            turb_data, turb_peaks, turb_props, delete_dat_range
        )

    turb_cand = [
        [
            peak,
            math.floor(turb_props["left_ips"][i]),
            math.ceil(turb_props["right_ips"][i]),
            turb_props["prominences"][i],
        ]
        for i, peak in enumerate(turb_peaks)
    ]

    # convert to dataframe
    cands_df = pd.DataFrame(turb_cand)

    # load ground truths
    truths = pd.read_csv(truths_filename)

    truths = truths[truths["label_of_peak"] != "NAP"]

    # drop all rows in cands that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df


def get_cands_turb_SKP(turb_filename, truths_filename, is_augmented=False):
    """
    get cands from turbidity skyrocketing peaks
    """
    # load in data
    if not is_augmented:
        turb_data = dm.read_in_preprocessed_timeseries(turb_filename)
    else:
        turb_data = np.array(dm.read_in_timeseries(turb_filename, True))

    # collect candidate peaks
    prominence_range = [20, None]  # higher than that of fDOM
    width_range = [None, None]
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        turb_data[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_df = pd.DataFrame(cands)

    # load in ground truths
    truths = pd.read_csv(truths_filename)

    # drop all non skp
    truths = truths[truths["label_of_peak"] != "NAP"]

    # drop all rows in cands that don't have a label
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df.columns = ["idx_of_peak", "left_base", "right_base", "amplitude"]

    return cands_df


def get_cands_turb_FPT(
    turb_filename, truths_filename, is_augmented=False, fpt_lookup_path=None
):
    """
    get cands from turbidity flat plateaus
    """
    # TODO: modify this to use the augmented file lookup for FPT peaks
    # TODO: look over this function as a whole, may not work well

    # load data
    if not is_augmented:
        turb_data = dm.read_in_preprocessed_timeseries(turb_filename)
    else:
        turb_data = np.array(dm.read_in_timeseries(turb_filename, True))

    cands = auxiliary_functions.detect_flat_plat(turb_data, 100, 40)

    turb_flat_plat_indxs = []
    for i in range(cands.shape[0]):
        if cands[i] == 1:
            turb_flat_plat_indxs.append(i)

    # create dataframe
    last_val = -1
    start_idx = -1
    end_idx = -1

    start_indices = []
    end_indices = []

    for idx, val in enumerate(turb_flat_plat_indxs):
        if val != last_val + 1:
            # we are now in a new peak, save stuff
            start_idx = val
            start_indices.append(start_idx)

            end_idx = last_val
            end_indices.append(end_idx)

        elif idx + 1 == len(turb_flat_plat_indxs):
            end_indices.append(val)

        # set last val
        last_val = val

    # drop first index in end indices
    del end_indices[0]

    cands = [[]]

    # need to get the actual prominence value
    # TODO: ensure the amplitude section is working as intended
    for i in range(len(start_indices)):
        cands.append(
            [
                start_indices[i],
                start_indices[i],
                end_indices[i],
                turb_data[start_indices[i]][1],
            ]
        )

    # create dataframe
    cands_df = pd.DataFrame(cands)

    # drop first row (incorrect val)
    cands_df = cands_df.drop([0])

    # load truths
    truths = pd.read_csv(truths_filename)

    truths = truths[truths["label_of_peak"] != "NAP"]

    # drop all rows in cands not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df.columns = ["idx_of_peak", "left_base", "right_base", "amplitude"]

    return cands_df


def get_cands_turb_NAP(
    turb_filename, truths_filename, is_augmented=False, delete_date_range=None
):
    """
    get all candidates from non anomaly peak data in turb
    """

    # load data
    if not is_augmented:
        turb_data = dm.read_in_preprocessed_timeseries(turb_filename)
    else:
        turb_data = np.array(dm.read_in_timeseries(turb_filename, True))

    turb_cand_params = {
        "prom": [6, None],
        "width": [None, None],
        "wlen": 200,
        "dist": 1,
        "rel_h": 0.6,
    }

    turb_peaks, turb_props = get_candidates(turb_data, turb_cand_params)

    if delete_date_range == None:
        delete_date_range = "Data/misc/flat_plat_ranges.txt"

    if not is_augmented:
        turb_peaks, turb_props = dp.delete_missing_data_peaks(
            turb_data, turb_peaks, turb_props, delete_date_range
        )

    turb_cand = [
        [
            peak,
            math.floor(turb_props["left_ips"][i]),
            math.ceil(turb_props["right_ips"][i]),
            turb_props["prominences"][i],
        ]
        for i, peak in enumerate(turb_peaks)
    ]

    # convert to dataframe
    cands_npp = pd.DataFrame(turb_cand)

    # collect candidate peaks
    prominence_range = [20, None]  # higher than that of fDOM
    width_range = [None, None]
    wlen = 100
    distance = 1
    rel_height = 0.6

    # Get list of all peaks that could possibly be plummeting peaks
    peaks, props = find_peaks(
        turb_data[:, 1],
        height=(None, None),
        threshold=(None, None),
        distance=distance,
        prominence=prominence_range,
        width=width_range,
        wlen=wlen,
        rel_height=rel_height,
    )

    # Form candidate set from returned information
    cands = [
        [
            peak,
            math.floor(props["left_ips"][i]),
            math.ceil(props["right_ips"][i]),
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_nskp = pd.DataFrame(cands)

    # load data
    if not is_augmented:
        turb_data = dm.read_in_preprocessed_timeseries(turb_filename)
    else:
        turb_data = np.array(dm.read_in_timeseries(turb_filename, True))

    cands = auxiliary_functions.detect_flat_plat(turb_data, 100, 40)

    turb_flat_plat_indxs = []
    for i in range(cands.shape[0]):
        if cands[i] == 1:
            turb_flat_plat_indxs.append(i)

    # create dataframe
    last_val = -1
    start_idx = -1
    end_idx = -1

    start_indices = []
    end_indices = []

    for idx, val in enumerate(turb_flat_plat_indxs):
        if val != last_val + 1:
            # we are now in a new peak, save stuff
            start_idx = val
            start_indices.append(start_idx)

            end_idx = last_val
            end_indices.append(end_idx)

        elif idx + 1 == len(turb_flat_plat_indxs):
            end_indices.append(val)

        # set last val
        last_val = val

    # drop first index in end indices
    del end_indices[0]

    cands = [[]]
    for i in range(len(start_indices)):
        cands.append([start_indices[i], start_indices[i], end_indices[i]])

    # create dataframe
    cands_nfpt = pd.DataFrame(cands)

    # drop first row (incorrect val)
    cands_nfpt = cands_nfpt.drop([0])

    # concat frames
    cands_df = pd.concat([cands_nskp, cands_npp, cands_nfpt])

    cands_df = cands_df.sort_values(by=[0], kind="stable")

    cands_df = cands_df.reset_index(drop=True)

    # import truths
    truths = pd.read_csv(truths_filename)

    truths = truths[truths["label_of_peak"] == "NAP"]

    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df
