import Tools.data_movement as dm
import Tools.data_processing as dp
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
import math
import pandas as pd
import copy

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
def get_cands_fDOM_PP():
    """
    Get candidates from fDOM phantom peaks
    """

    # pass fDOM data through low pass filter
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )
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

    cands_df = pd.DataFrame(cands)

    # now load in ground truths, and drop all things in cands that are not anomaly peaks
    # Import ground truth values
    truth_fname = "../Data/labeled_data/ground_truths/fDOM/fDOM_PP/julian_time/fDOM_PP_0k-300k.csv"

    truths = pd.read_csv(truth_fname)

    # drop all NPP indices
    truths = truths[truths["label_of_peak"] != "NPP"]

    # drop all rows in cnads that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # reindex frame
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    # return frame
    return cands_df


def get_cands_fDOM_SKP():
    """
    Get candidates from fDOM Skyrocketing peaks
    """

    # load in fdom data
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )

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
    truth_fname = "../Data/labeled_data/ground_truths/fDOM/fDOM_SKP/julian_time/fDOM_SKP_0k-300k.csv"

    truths = pd.read_csv(truth_fname)

    # drop all NSKP indices
    truths = truths[truths["label_of_peak"] != "NSKP"]

    # drop all rows in cands that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df


def get_cands_fDOM_PLP():
    """
    Get candidates from fDOM plummeting peaks
    """

    # load data
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )

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
            props["prominences"][i],
        ]
        for i, peak in enumerate(peaks)
    ]

    cands_df = pd.DataFrame(cands)

    # get truths
    truth_fname = "../Data/labeled_data/ground_truths/fDOM/fDOM_PLP/julian_time/fDOM_PLP_0k-300k.csv"
    truths = pd.read_csv(truth_fname)

    # drop all NPLP indices
    truths = truths[truths["label_of_peak"] != "NPLP"]

    # drop all rows in cands not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    # return data
    return cands_df


def get_cands_fDOM_NAP():
    """
    get candidates from non anomaly peak data in fdom
    """

    # get all peaks from other data types, drop dupes, go from there
    # get peaks from NPP:
    # pass fDOM data through low pass filter
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )
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
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )

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
    fDOM_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv"
    )

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

    # concat dataframes
    cands_df = pd.concat([cands_nskp, cands_npp, cands_nplp])
    cands_df = cands_df.sort_values(by=[0], kind="stable")
    # cands_df = cands_df[~cands_df.index.duplicated(keep='first')]
    cands_df = cands_df.reset_index(drop=True)

    # import ground truths
    truth_fname = "../Data/labeled_data/ground_truths/fDOM/fDOM_all_julian_0k-300k.csv"
    truths = pd.read_csv(truth_fname)

    truths = truths[truths["label_of_peak"] == "NAP"]

    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df


######## TURBIDITY #########
def get_cands_turb_PP():
    """
    Get candidates from turbidity phantom peaks
    """

    # load data
    turb_data = dm.read_in_preprocessed_timeseries(
        "../Data/converted_data/julian_format/turbidity_raw_10.1.2011_9.4.2020.csv"
    )

    turb_cand_params = {
        "prom": [6, None],
        "width": [None, None],
        "wlen": 200,
        "dist": 1,
        "rel_h": 0.6,
    }

    turb_peaks, turb_props = get_candidates(turb_data, turb_cand_params)

    turb_peaks, turb_props = dp.delete_missing_data_peaks(
        turb_data, turb_peaks, turb_props, "../Data/misc/flat_plat_ranges.txt"
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
    truth_fname = "../Data/labeled_data/ground_truths/turb/turb_pp/julian_time/turb_pp_0k-300k_labeled"
    truths = pd.read_csv(truth_fname)

    truths = truths[truths["label_of_peak"] != "NPP"]

    # drop all rows in cands that are not in truths
    cands_df = cands_df[cands_df[0].isin(truths["idx_of_peak"])]

    # rename cols
    cands_df = cands_df.rename(
        columns={0: "idx_of_peak", 1: "left_base", 2: "right_base", 3: "amplitude"}
    )

    return cands_df
