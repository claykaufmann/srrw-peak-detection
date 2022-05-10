"""
This module contains helper functions for data augmentation
"""
import math
import sys

# so we can import tools module
import Tools.data_processing as dp
from datetime import timedelta
import pandas as pd
import numpy as np
import copy
import random
from scipy import interpolate


def next_time_entry(current_entry: float) -> float:
    """
    This function returns the next time entry in julian time

    current_entry: a julina time float

    return: julian time + 15 minutes from past julian time
    """

    # convert julian to datetime
    date_time_init = dp.julian_to_datetime(current_entry)

    # find next date time (add 15 minutes)
    next_entry = date_time_init + timedelta(minutes=15)

    # convert date time to julian time
    final_julian_time = dp.datetime_to_julian(next_entry)

    # return julian time
    return final_julian_time


def convert_df_julian_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a dataframe julian timestamp to datetime ISO 8601 format

    df: a dataframe

    return: changed dataframe
    """
    # iterate over dataframe, replacing timestamp vals
    for i, row in df.iterrows():
        df.loc[i, "timestamp"] = dp.julian_to_datetime(
            df.loc[i, "timestamp"]
        ).isoformat()

        # add stupid 0.00Z to fit trainset format
        df.loc[i, "timestamp"] = df.loc[i, "timestamp"] + ".000Z"

    return df


def get_last_augment_index(dataframe) -> int:
    """
    Collects the last index of the augmented time series

    dataframe: the dataframe of index to check

    returns: the last index
    """
    return dataframe.shape[0]


def get_ends_of_peak(cands_df: pd.DataFrame, peak_index) -> tuple():
    """
    get left and right ends of a peak from the respective dataframes

    cands_df: the candidate dataframe

    peak_index: the index of the peak

    returns: the left and right base of the peak segment
    """
    # use cands_df to return left and right base of peak index
    new_cands = copy.deepcopy(cands_df)
    new_cands = new_cands.loc[new_cands["idx_of_peak"] == peak_index]

    left_base = new_cands["left_base"]
    right_base = new_cands["right_base"]

    left_base = left_base.to_list()
    left_base = left_base[0]

    right_base = right_base.to_list()
    right_base = right_base[0]

    # return left and right
    return left_base, right_base


# might actually work if correct indices are just passed in...
def build_temp_dataframes(
    fdom, stage, turb, prev, next, fdom_idx, stage_idx, turb_idx
) -> tuple():
    """
    build the temporary dataframes for the peak segment

    fdom: the fdom main dataframe

    stage: the stage dataframe

    turb: the turb dataframe

    prev: how far back the peak goes (index wise)

    next: how far forward the peak goes (index wise)

    fdom_idx: the peak index for fdom

    stage_idx: the relevant stage index

    turb_idx: the relevant turb index

    returns: the new time segments for each dataframe
    """
    fDOM_raw_time_range = pd.DataFrame(fdom.iloc[fdom_idx - prev : fdom_idx + next + 1])

    # get stage data range
    stage_time_range = pd.DataFrame(stage.iloc[stage_idx - prev : stage_idx + next + 1])

    # get turbidity data range
    turb_time_range = pd.DataFrame(turb.iloc[turb_idx - prev : turb_idx + next + 1])

    new_fdom = copy.deepcopy(fDOM_raw_time_range)
    new_stage = copy.deepcopy(stage_time_range)
    new_turb_raw = copy.deepcopy(turb_time_range)

    return new_fdom, new_stage, new_turb_raw


def widen_augment(df, peak_idx) -> pd.DataFrame:
    """
    widen the peaks by increasing values across the peak by a set noise val

    df: the dataframe to augment

    peak_idx: the index of the peak

    returns: The augmented dataframe
    """
    mu, sigma = 0.001, 0.1
    noise = np.random.normal(mu, sigma, df.shape[0])

    df.loc[:, "value"] = df.loc[:, "value"] + noise

    return df


def heighten_augment(
    df, peak_idx, lower_bound_multiplier, upper_bound_multiplier
) -> pd.DataFrame:
    """
    heighten the main peak ampltiude

    df: the dataframe to augment

    peak_index: the index of the peak

    lower_bound_multiplier, upper_bound_multiplier: The bounds for which peak can be multiplied

    returns: the augmented dataframe
    """
    # gen a random number to multiply amplitude by
    random_val = random.uniform(
        lower_bound_multiplier,
        upper_bound_multiplier,
    )
    new_peak_val = df.loc[peak_idx, "value"] * (1 + random_val)
    df.loc[peak_idx, "value"] = new_peak_val

    return df


def augment_data(df, peak_index, lower_bound_multiplier, upper_bound_multiplier):
    """
    augment the given dataframe
    decides whether to heighten or widen the peak

    df: dataframe to augment

    peak_index: index of the peak

    lower_bound_multiplier, upper_bound_multiplier: The bounds for which peak can be multiplied

    returns: the augmented dataframe
    """
    # gen random number
    random.seed()
    widen_or_heighten = random.randint(0, 1)

    # if 0, widen peak, else heighten
    if widen_or_heighten == 0:
        df = widen_augment(df, peak_index)

    else:
        df = heighten_augment(
            df, peak_index, lower_bound_multiplier, upper_bound_multiplier
        )

    # return augmented dataframe
    return df


def concat_dataframes(
    main_df_labeled,
    main_fdom,
    main_stage,
    main_turb,
    new_fdom,
    new_stage,
    new_turb,
    new_label,
):
    """
    Concatenate the augmented dataframes

    main_df_labeled: the labeled dataframe that holds all augmented data

    main_fdom: the main fdom dataframe being concatenated

    main_stage: the main stage df being concatenated

    main_turb: the main turb df being concatenated

    new fdom, stage, turb: the small peak range df's holding newly augmented data

    new_label: the label for the relevant data that was augmented (either fdom or turb)

    returns: the main augmented dataframes
    """
    main_df_labeled = pd.concat([main_df_labeled, new_label])
    main_fdom = pd.concat([main_fdom, new_fdom], ignore_index=True)
    main_stage = pd.concat([main_stage, new_stage], ignore_index=True)
    main_turb = pd.concat([main_turb, new_turb], ignore_index=True)

    return main_df_labeled, main_fdom, main_stage, main_turb


def update_dataframes(
    prev_time_entry,
    df,
    peak_index,
    prev_dist,
    main_augment_df,
    stage_df,
    non_augment_df,
    label_of_peak,
):
    """
    set updated timestamps, peak values, etc.

    prev_time_entry is the past time entry

    df is the main dataframe being actually augmented (fdom or turb)

    peak_index is the index of the augmented peak

    prev_dist is how far back from the peak the main df being augmented goes

    main_augment_df is the overall df that holds the augmented data

    stage_df is the stage section for df

    non_augment_df is either the turb section or fdom section, depending on if fdom or turb is being augmented

    label_of_peak is the peak label

    returns: the new label, and the augmented dataframes with the updated timestamps
    """
    # set new time entry to be the prev entry passed into this function
    new_time_entry = prev_time_entry

    # add temp col to access indices
    df["tmp"] = df.index
    for i, row in df.iterrows():
        # get new timestamp
        new_time_entry = next_time_entry(new_time_entry)

        # check timestamps
        if df.loc[i, "tmp"] == peak_index:
            # get the new index of the peak
            # this is for the new label for the labeled augmented data
            new_peak_index = get_last_augment_index(main_augment_df) + prev_dist
            new_peak_timestamp = new_time_entry
            new_peak_val = df.loc[i, "value"]

        # update timestamps of df's
        df.loc[i, "timestamp"] = new_time_entry
        stage_df.loc[i, "timestamp"] = new_time_entry
        non_augment_df.loc[i, "timestamp"] = new_time_entry

    # delete the extra label
    del df["tmp"]

    # create new label for labeled data
    new_label = pd.DataFrame(
        [[new_peak_timestamp, new_peak_val, label_of_peak, new_peak_index]],
        columns=["timestamp_of_peak", "value_of_peak", "label_of_peak", "idx_of_peak"],
    )

    # return the label, and return all new dataframes
    return new_label, df, stage_df, non_augment_df, new_time_entry


def smooth_data(
    main_fdom,
    main_stage,
    main_turb,
    prev_added_entry,
    smooth_lower_bound,
    smooth_upper_bound,
    fdom_val,
    turb_val,
    stage_val,
):
    """
    create a dataframe of data smoothing to smooth data in between sections of anomaly data

    main_fdom, main_stage, main_turb: the main augmented dataframes for fdom, stage, and turb, the ones that all data is concatenated into

    new_fdom, new_stage, new_turb: the newly augmented dataframes within the small peak range

    prev_added_entry: the previous timestamp, used to sync time up in the main dataframes

    smooth_lower_bound, smooth_upper_bound: The range of datapoints over which the data will be smoothed

    returns: The main dataframes with augmented data including the smoothed data
    """
    # read prev_added_entry into the new_time_entry
    new_time_entry = prev_added_entry

    # get the last values of fdom, stage, and turb
    last_fdom = main_fdom.loc[main_fdom.shape[0] - 1, "value"]
    last_stage = main_stage.loc[main_stage.shape[0] - 1, "value"]
    last_turb = main_turb.loc[main_turb.shape[0] - 1, "value"]

    # create list of x points the data needs to pass through
    # these are timestamps
    # randomly generate the number we need, between 12-24 hours
    random.seed()

    number_of_points = random.randrange(smooth_lower_bound, smooth_upper_bound)

    # decrease using a fifth of the number of points
    slope_number = number_of_points // 7

    # save leftover length
    left_over_length = number_of_points - slope_number

    x_points = []
    new_time_entry = prev_added_entry
    for x in range(slope_number):
        # create list of time points
        new_time_entry = next_time_entry(new_time_entry)
        x_points.append(new_time_entry)

    x = np.array(x_points)

    # remove neg numbers, to stop log issues
    if last_fdom < 0:
        last_fdom = 0.000001
    if last_stage < 0:
        last_stage = 0.000001
    if last_turb < 0:
        last_turb = 0.000001

    fdom_points = np.geomspace(last_fdom, fdom_val, slope_number)
    stage_points = np.geomspace(last_stage, stage_val, slope_number)
    turb_points = np.geomspace(last_turb, turb_val, slope_number)

    # create interpolation vals
    fdom_interpolation = interpolate.interp1d(x, fdom_points)
    stage_interpolation = interpolate.interp1d(x, stage_points)
    turb_interpolation = interpolate.interp1d(x, turb_points)

    # interpolated values
    fdom_new = fdom_interpolation(x)
    stage_new = stage_interpolation(x)
    turb_new = turb_interpolation(x)

    # make dataframes out of these
    fdom_data = {"timestamp": x, "value": fdom_new}
    fdom_df = pd.DataFrame(data=fdom_data)

    stage_data = {"timestamp": x, "value": stage_new}
    stage_df = pd.DataFrame(data=stage_data)

    turb_data = {"timestamp": x, "value": turb_new}
    turb_df = pd.DataFrame(data=turb_data)

    # concat dataframes
    main_fdom = pd.concat([main_fdom, fdom_df], ignore_index=True)
    main_stage = pd.concat([main_stage, stage_df], ignore_index=True)
    main_turb = pd.concat([main_turb, turb_df], ignore_index=True)

    # add in noise vals around the flat val
    mu, sigma = 0.00001, 0.0001
    noise = np.random.normal(mu, sigma, left_over_length)
    fdom_noise = noise + fdom_val

    mu, sigma = 0.00001, 0.0001
    noise = np.random.normal(mu, sigma, left_over_length)
    stage_noise = noise + stage_val

    mu, sigma = 0.00001, 0.0001
    noise = np.random.normal(mu, sigma, left_over_length)
    turb_noise = noise + turb_val

    # create new set of timestamps
    flat_x_points = []
    for x in range(left_over_length):
        new_time_entry = next_time_entry(new_time_entry)
        flat_x_points.append(new_time_entry)
    x = np.array(flat_x_points)

    # gen temp dataframes
    flat_fdom = {"timestamp": x, "value": fdom_noise}
    flat_fdom_df = pd.DataFrame(data=flat_fdom)

    flat_stage = {"timestamp": x, "value": stage_noise}
    flat_stage_df = pd.DataFrame(data=flat_stage)

    flat_turb = {"timestamp": x, "value": turb_noise}
    flat_turb_df = pd.DataFrame(data=flat_turb)

    # concat flat section
    main_fdom = pd.concat([main_fdom, flat_fdom_df], ignore_index=True)
    main_stage = pd.concat([main_stage, flat_stage_df], ignore_index=True)
    main_turb = pd.concat([main_turb, flat_turb_df], ignore_index=True)

    return main_fdom, main_stage, main_turb, new_time_entry


def write_augmented_data_to_csv(
    labeled_fdom_path,
    unlabeled_fdom_path,
    labeled_turb_path,
    unlabeled_turb_path,
    augmented_fDOM_labeled,
    augmented_fDOM_raw,
    augmented_turb_raw_fdom,
    augmented_stage_raw_fdom,
    augmented_turb_labeled,
    augmented_turb_raw,
    augmented_fDOM_raw_turb,
    augmented_stage_raw_turb,
):
    # call to_csv for each dataframe
    # for each dataframe, we also drop the index

    # write fDOM augmented data
    augmented_fDOM_labeled.to_csv(
        labeled_fdom_path + "labeled_fdom_peaks.csv", index=False
    )
    augmented_fDOM_raw.to_csv(
        unlabeled_fdom_path + "unlabeled_fdom.csv", index=False, header=False
    )
    augmented_turb_raw_fdom.to_csv(
        unlabeled_fdom_path + "unlabeled_turb.csv", index=False, header=False
    )
    augmented_stage_raw_fdom.to_csv(
        unlabeled_fdom_path + "unlabeled_stage.csv", index=False, header=False
    )

    # write turb augmented data
    augmented_turb_labeled.to_csv(
        labeled_turb_path + "labeled_turb_peaks.csv", index=False
    )
    augmented_turb_raw.to_csv(
        unlabeled_turb_path + "unlabeled_turb.csv", index=False, header=False
    )
    augmented_fDOM_raw_turb.to_csv(
        unlabeled_turb_path + "unlabeled_fdom.csv", index=False, header=False
    )
    augmented_stage_raw_turb.to_csv(
        unlabeled_turb_path + "unlabeled_stage.csv", index=False, header=False
    )


def write_to_trainset_csv(
    augmented_fDOM_raw,
    augmented_turb_raw_fdom,
    augmented_stage_raw_fdom,
    trainset_fdom_path,
    augmented_turb_raw,
    augmented_fDOM_raw_turb,
    augmented_stage_raw_turb,
    trainset_turb_path,
):
    # TODO: add peak labels in

    # start by creating a dataframe that has the correct columns
    trainset_fdom_df = pd.DataFrame(columns=["series", "timestamp", "value", "label"])
    trainset_turb_df = pd.DataFrame(columns=["series", "timestamp", "value", "label"])

    # ~~~~~~~ fDOM section ~~~~~~~
    # start by just adding the fDOM data into the series, need to replace all timestamps
    fdom_trainset_raw = copy.deepcopy(augmented_fDOM_raw)
    fdom_turb_trainset_raw = copy.deepcopy(augmented_turb_raw_fdom)
    fdom_stage_trainset_raw = copy.deepcopy(augmented_stage_raw_fdom)

    # convert timestamps to julian
    fdom_trainset_raw = convert_df_julian_to_datetime(fdom_trainset_raw)
    fdom_turb_trainset_raw = convert_df_julian_to_datetime(fdom_turb_trainset_raw)
    fdom_stage_trainset_raw = convert_df_julian_to_datetime(fdom_stage_trainset_raw)

    # add in new values
    fdom_trainset_raw["series"] = "fDOM"
    fdom_trainset_raw["label"] = ""

    fdom_turb_trainset_raw["series"] = "turb"
    fdom_turb_trainset_raw["label"] = ""

    fdom_stage_trainset_raw["series"] = "stage"
    fdom_stage_trainset_raw["label"] = ""

    # reorder columns
    fdom_trainset_raw = fdom_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )
    fdom_turb_trainset_raw = fdom_turb_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )
    fdom_stage_trainset_raw = fdom_stage_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )

    # concat into single dataframe
    trainset_fdom_df = pd.concat(
        [fdom_trainset_raw, fdom_turb_trainset_raw, fdom_stage_trainset_raw]
    )

    # sort together
    trainset_fdom_df = trainset_fdom_df.sort_values(by=["timestamp"], kind="stable")

    # export to csv
    trainset_fdom_df.to_csv(trainset_fdom_path + "fdom_augmented.csv", index=False)

    # ~~~~~~~ turbidity section ~~~~~~~

    # create new dataframes
    turb_trainset_raw = copy.deepcopy(augmented_turb_raw)
    turb_fdom_trainset_raw = copy.deepcopy(augmented_fDOM_raw_turb)
    turb_stage_trainset_raw = copy.deepcopy(augmented_stage_raw_turb)

    # convert timestamps
    turb_trainset_raw = convert_df_julian_to_datetime(turb_trainset_raw)
    turb_fdom_trainset_raw = convert_df_julian_to_datetime(turb_fdom_trainset_raw)
    turb_stage_trainset_raw = convert_df_julian_to_datetime(turb_stage_trainset_raw)

    # add in new values
    turb_fdom_trainset_raw["series"] = "fDOM"
    turb_fdom_trainset_raw["label"] = ""

    turb_trainset_raw["series"] = "turb"
    turb_trainset_raw["label"] = ""

    turb_stage_trainset_raw["series"] = "stage"
    turb_stage_trainset_raw["label"] = ""

    # reorder columns
    turb_fdom_trainset_raw = turb_fdom_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )
    turb_trainset_raw = turb_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )
    turb_stage_trainset_raw = turb_stage_trainset_raw.reindex(
        columns=["series", "timestamp", "value", "label"]
    )

    # concat into single dataframe
    trainset_turb_df = pd.concat(
        [turb_fdom_trainset_raw, turb_trainset_raw, turb_stage_trainset_raw]
    )

    # sort together
    trainset_turb_df = trainset_turb_df.sort_values(by=["timestamp"], kind="stable")

    # export to csv
    trainset_turb_df.to_csv(trainset_turb_path + "turb_augmented.csv", index=False)


def check_class_balance(current_balance: dict, class_labels: list) -> list:
    """
    check the current balance of classes
    """
    # check the current balances, and then return which peak needs to be augmented
    lowest_balance = []

    curr_min = math.inf
    for key in current_balance.keys():
        count = current_balance[key]
        if count <= curr_min:
            curr_min = count
            lowest_balance.append((key, count))

    # iterate over curr balance again to find
    final_label_list = []
    for i in range(len(lowest_balance)):
        # find all keys with curr min
        if lowest_balance[i][1] == curr_min:
            final_label_list.append(lowest_balance[i][0])

    # return a random item from the list
    return random.choice(final_label_list)
