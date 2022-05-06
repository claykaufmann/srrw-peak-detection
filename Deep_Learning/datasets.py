"""
this file contains the datasets for the deep learning section of the project
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import copy
from Tools import augmentation_helpers as augment
from Tools import data_movement as dm
from Tools import data_processing as dp
from Tools.get_all_cands import (
    get_all_cands_fDOM,
    get_all_cands_turb,
    get_all_truths,
    get_all_truths_fDOM,
)


class fdomDataset(data.Dataset):
    """
    the fdom dataset
    """

    def __init__(
        self,
        labeler,
        fdom_data_raw_dir,
        stage_raw_dir,
        turb_raw_dir,
        fdom_labeled_raw_dir,
        fdom_augmented_dir=None,
        stage_augmented_dir=None,
        turb_augmented_dir=None,
        fdom_labeled_aug_dir=None,
        fpt_lookup_filename=None,
        fsk_lookup_filename=None,
        window_size=15,
    ) -> None:
        """
        constructor

        PARAMS:
        labeler: a sklearn LabelEncoder, for labeling string classes into numbers
        fdom_data_raw_dir: path to raw fdom data
        stage_raw_dir: path to raw stage data
        turb_raw_dir: path to raw turb data
        fdom_labeled_raw_dir: path to labeled raw fdom data
        fdom_augmented_dir: path to augmented fdom data
        stage_augmented_dir: path to augmented stage data
        turb_augmented_dir: path to augmented turb data
        fdom_labeled_aug_dir: path to augmented labeled fdom data
        window_size: the width of data to use for each sample, note this is the distance before and after the main peak index
        """

        super(fdomDataset, self).__init__()

        self.label_encoder = labeler

        self.fdom_raw_path = fdom_data_raw_dir
        self.stage_raw_path = stage_raw_dir
        self.turb_raw_path = turb_raw_dir

        # labeled data
        self.fdom_raw_labeled_path = fdom_labeled_raw_dir

        # augmented paths
        self.fdom_aug_path = fdom_augmented_dir
        self.stage_aug_path = stage_augmented_dir
        self.turb_aug_path = turb_augmented_dir

        # labeled data
        self.fdom_aug_labeled_path = fdom_labeled_aug_dir

        self.fpt_lookup_filename = fpt_lookup_filename
        self.fsk_lookup_filename = fsk_lookup_filename

        # generate the dataset
        self.get_data(window_size)

        print(f"{len(self.data)} candidates found.")

    def get_data(self, window_size):
        """
        saves the actual data into the dataset

        PARAMS:
        window_size: the size of each segment (length, in time-series data)
        """

        # load all datasets
        # load non augmented data
        # raw data
        fdom_raw = dm.read_in_preprocessed_timeseries(self.fdom_raw_path)
        stage_raw = dp.align_stage_to_fDOM(
            fdom_raw, dm.read_in_preprocessed_timeseries(self.stage_raw_path)
        )
        turb_raw = dm.read_in_preprocessed_timeseries(self.turb_raw_path)

        # get cands from non augmented data
        peaks = get_all_cands_fDOM(self.fdom_raw_path, self.fdom_raw_labeled_path)

        # get all respective truths
        truths = get_all_truths_fDOM(self.fdom_raw_labeled_path)

        # load in augmented data
        if self.fdom_aug_path is not None:
            # use normal timeseries, as we are not cutting out specific data
            fdom_aug = np.array(dm.read_in_timeseries(self.fdom_aug_path, True))
            stage_aug = np.array(dm.read_in_timeseries(self.stage_aug_path, True))
            turb_aug = np.array(dm.read_in_timeseries(self.turb_aug_path, True))

            # concatenate arrays of raw data
            fdom_raw = np.concatenate([fdom_raw, fdom_aug])
            stage_raw = np.concatenate([stage_raw, stage_aug])
            turb_raw = np.concatenate([turb_raw, turb_aug])

            # get all cands from augmented data, augment cands
            aug_peaks = get_all_cands_fDOM(
                self.fdom_aug_path,
                self.fdom_aug_labeled_path,
                True,
                self.fpt_lookup_filename,
                self.fsk_lookup_filename,
            )

            # get all truths
            aug_truths = get_all_truths_fDOM(self.fdom_aug_labeled_path, True)

            # concat these two frames
            peaks = pd.concat([peaks, aug_peaks])
            truths = pd.concat([truths, aug_truths])

        # update raw datasets to only have necessary data
        time = copy.deepcopy(fdom_raw)[:, 0]
        fdom_raw = fdom_raw[:, 1]
        stage_raw = stage_raw[:, 1]
        turb_raw = turb_raw[:, 1]

        # initiate arrays for samples and labels that we will read data into
        X = []
        y = []

        for i, peak in peaks.iterrows():
            # get start and end indices
            peak_idx = int(peak["idx_of_peak"])
            left_base = int(peak["left_base"])
            right_base = int(peak["right_base"])

            left = abs(peak_idx - left_base)
            right = abs(peak_idx - right_base)

            # use these indices to collect the data for stage and turb
            # each sample follows this order: 0 = fdom, 1 = stage, 2 = turb, 3 = time
            sample = [
                fdom_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                stage_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                turb_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                time[peak_idx - left : peak_idx + right + 1].tolist(),
            ]

            # sometimes the sample doesn't actually include any data, ensure it has the correct size
            if len(sample[0]) > 0:

                X.append(sample)

                # get label
                label = truths.loc[
                    truths["idx_of_peak"] == peak_idx, "label_of_peak"
                ].iloc[0]

                # convert label to normalized integer value, using passed in label encoder
                label = self.label_encoder.transform([label])

                y.append(label)
            else:
                # if sample has incorrect shape, don't add it
                print("WARNING: shape of a sample is incorrect, not adding it")
                print(f"Error shape is: {sample.shape}")
                print(f"Error vals: {fdom_raw[peak_idx - left : peak_idx + right + 1]}")

        # assert that X and y are the same length, so we have a label for each data point
        assert len(X) == len(y)

        # save data and truths
        self.data = X
        self.truths = y

    def __len__(self):
        """
        returns len of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        returns the sample and label at index

        RETURNS A SAMPLE WHERE:
        sample[0] = raw data
            sample[0][0] = fdom raw data
            sample[0][1] = stage raw data
            sample[0][2] = turb raw data
        sample[1] = int, the encoded label for the data, encoded with passed in encoder
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # collect the sample, its a vector of fdom, stage, and turb
        sample = (self.data[idx], self.truths[idx])

        # return the sample
        return sample


class fdomAugOnlyDataset(data.Dataset):
    """
    the fdom dataset for only augmented data

    used for training with just augmented, class balanced data
    """

    def __init__(
        self,
        labeler,
        fdom_augmented_dir,
        stage_augmented_dir,
        turb_augmented_dir,
        fdom_labeled_aug_dir,
        fpt_lookup_filename,
        fsk_lookup_filename,
        window_size=15,
    ) -> None:
        """
        constructor

        PARAMS:
        labeler: a sklearn LabelEncoder, for labeling string classes into numbers
        fdom_data_raw_dir: path to raw fdom data
        stage_raw_dir: path to raw stage data
        turb_raw_dir: path to raw turb data
        fdom_labeled_raw_dir: path to labeled raw fdom data
        fdom_augmented_dir: path to augmented fdom data
        stage_augmented_dir: path to augmented stage data
        turb_augmented_dir: path to augmented turb data
        fdom_labeled_aug_dir: path to augmented labeled fdom data
        window_size: the width of data to use for each sample, note this is the distance before and after the main peak index
        """

        super(fdomAugOnlyDataset, self).__init__()

        self.label_encoder = labeler

        # augmented paths
        self.fdom_aug_path = fdom_augmented_dir
        self.stage_aug_path = stage_augmented_dir
        self.turb_aug_path = turb_augmented_dir

        # labeled data
        self.fdom_aug_labeled_path = fdom_labeled_aug_dir

        self.fpt_lookup_filename = fpt_lookup_filename
        self.fsk_lookup_filename = fsk_lookup_filename

        # generate the dataset
        self.get_data(window_size)

        print(f"{len(self.data)} candidates found.")

    def get_data(self, window_size):
        """
        saves the actual data into the dataset

        PARAMS:
        window_size: the size of each segment (length, in time-series data)
        """
        # use normal timeseries, as we are not cutting out specific data
        # indices select only the second row, which are the respective values
        fdom_raw = np.array(dm.read_in_timeseries(self.fdom_aug_path, True))
        stage_raw = np.array(dm.read_in_timeseries(self.stage_aug_path, True))[:, 1]
        turb_raw = np.array(dm.read_in_timeseries(self.turb_aug_path, True))[:, 1]

        # get the time indices
        time = copy.deepcopy(fdom_raw)
        time = time[:, 0]

        # make fdom just the values
        fdom_raw = fdom_raw[:, 1]

        # get all cands from augmented data, augment cands
        peaks = get_all_cands_fDOM(
            self.fdom_aug_path,
            self.fdom_aug_labeled_path,
            True,
            self.fpt_lookup_filename,
            self.fsk_lookup_filename,
        )

        # get all truths
        truths = get_all_truths_fDOM(self.fdom_aug_labeled_path, True)

        # initiate arrays for samples and labels that we will read data into
        X = []
        y = []

        for i, peak in peaks.iterrows():
            # get start and end indices
            peak_idx = int(peak["idx_of_peak"])
            left_base = int(peak["left_base"])
            right_base = int(peak["right_base"])

            left = abs(peak_idx - left_base)
            right = abs(peak_idx - right_base)

            # use these indices to collect the data for stage and turb
            # each sample follows this order: 0 = fdom, 1 = stage, 2 = turb, 3 = time
            # need to modify how we are combining these, because it does not seem to be outputting the correct shape
            # shape should be [*, 4] (where * is variable length), instead it is [3, *, 2] for some reason
            sample = [
                fdom_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                stage_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                turb_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                time[peak_idx - left : peak_idx + right + 1].tolist(),
            ]

            # if a sample is zero length, dont add it
            if len(sample[0]) > 0:
                X.append(sample)

                # get label
                label = truths.loc[
                    truths["idx_of_peak"] == peak_idx, "label_of_peak"
                ].iloc[0]

                # convert label to normalized integer value, using passed in label encoder
                label = self.label_encoder.transform([label])

                y.append(label)
            else:
                print("WARNING: shape of a sample is zero, not adding it")
                print(f"Error shape is: {sample.shape}")
                print(f"Error vals: {fdom_raw[peak_idx - left : peak_idx + right + 1]}")

        # assert that X and y are the same length, so we have a label for each data point
        assert len(X) == len(y)

        # save data and truths
        self.data = X
        self.truths = y

    def __len__(self):
        """
        returns len of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        returns the sample and label at index

        RETURNS A SAMPLE WHERE:
        sample[0] = raw data
            sample[0][0] = fdom raw data
            sample[0][1] = stage raw data
            sample[0][2] = turb raw data
        sample[1] = int, the encoded label for the data, encoded with passed in encoder
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # collect the sample, its a vector of fdom, stage, and turb
        sample = (self.data[idx], self.truths[idx])

        # return the sample
        return sample


class turbidityDataset(data.Dataset):
    """
    turbidity dataset
    """

    def __init__(
        self,
        labeler,
        fdom_data_raw_dir,
        stage_raw_dir,
        turb_raw_dir,
        turb_labeled_raw_dir,
        fdom_augmented_dir=None,
        stage_augmented_dir=None,
        turb_augmented_dir=None,
        turb_labeled_aug_dir=None,
        window_size=15,
    ) -> None:
        """
        constructor

        PARAMS:
        labeler: a sklearn LabelEncoder, for labeling string classes into numbers
        fdom_data_raw_dir: path to raw fdom data
        stage_raw_dir: path to raw stage data
        turb_raw_dir: path to raw turb data
        turb_labeled_raw_dir: path to labeled turb data
        fdom_augmented_dir: path to augmented fdom data
        stage_augmented_dir: path to augmented stage data
        turb_augmented_dir: path to augmented turb data
        turb_labeled_aug_dir: path to augmented labeled turb data
        window_size: the width of data to use for each sample, note this is the distance before and after the main peak index
        """
        super(turbidityDataset, self).__init__()

        # save passed in vars
        self.label_encoder = labeler

        # raw data
        self.fdom_raw_path = fdom_data_raw_dir
        self.stage_raw_path = stage_raw_dir
        self.turb_raw_path = turb_raw_dir

        # augmented raw data
        self.fdom_aug_path = fdom_augmented_dir
        self.stage_aug_path = stage_augmented_dir
        self.turb_aug_path = turb_augmented_dir

        # labeled data
        self.turb_raw_labeled_path = turb_labeled_raw_dir
        self.turb_aug_labeled_path = turb_labeled_aug_dir

        # generate the dataset
        self.get_data(window_size)

        print(f"{len(self.data)} candidates found in test dataset.")

    def get_data(self, window_size):
        """
        collect candidates and truths, and save to dataset

        PARAMS:
        window_size: the size of each segment (length, in time-series data)
        """

        # load all datasets

        # load non augmented data
        fdom_raw = dm.read_in_preprocessed_timeseries(self.fdom_raw_path)
        stage_raw = dp.align_stage_to_fDOM(
            fdom_raw, dm.read_in_preprocessed_timeseries(self.stage_raw_path)
        )
        turb_raw = dm.read_in_preprocessed_timeseries(self.turb_raw_path)

        # get cands from non augmented data
        peaks = get_all_cands_turb(self.turb_raw_path, self.turb_raw_labeled_path)

        # get truths for said peaks
        truths = get_all_truths(self.turb_raw_labeled_path)

        # load augmented data
        if self.turb_aug_path is not None:
            # use normal timeseries, as we are not cutting out specific data
            fdom_aug = np.array(dm.read_in_timeseries(self.fdom_aug_path, True))
            stage_aug = np.array(dm.read_in_timeseries(self.stage_aug_path, True))
            turb_aug = np.array(dm.read_in_timeseries(self.turb_aug_path, True))

            # concatenate arrays of raw data
            fdom_raw = np.concatenate([fdom_raw, fdom_aug])
            stage_raw = np.concatenate([stage_raw, stage_aug])
            turb_raw = np.concatenate([turb_raw, turb_aug])

            # get peaks
            aug_peaks = get_all_cands_turb(
                self.turb_aug_path, self.turb_aug_labeled_path, True
            )

            # get all truths
            aug_truths = get_all_truths(self.turb_aug_labeled_path, True)

            # concat the augmented and raw peaks/truths
            peaks = pd.concat([peaks, aug_peaks])
            truths = pd.concat([truths, aug_truths])

        # get time, remove time from other datasets
        time = copy.deepcopy(turb_raw)[:, 0]
        turb_raw = turb_raw[:, 1]
        stage_raw = stage_raw[:, 1]
        fdom_raw = fdom_raw[:, 1]

        # instantiate arrays to load values into to be saved
        X = []
        Y = []

        for i, peak in peaks.iterrows():
            # get start and end indices
            peak_idx = int(peak["idx_of_peak"])
            left_base = int(peak["left_base"])
            right_base = int(peak["right_base"])

            left = abs(peak_idx - left_base)
            right = abs(peak_idx - right_base)

            # each sample follows this order: 0 = fdom, 1 = stage, 2 = turb, 3 = time
            sample = [
                fdom_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                stage_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                turb_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                time[peak_idx - left : peak_idx + right + 1].tolist(),
            ]

            if len(sample[0]) > 0:
                X.append(sample)

                # get label
                label = truths.loc[
                    truths["idx_of_peak"] == peak_idx, "label_of_peak"
                ].iloc[0]

                # convert label to normalized integer value, using passed in label encoder
                label = self.label_encoder.transform([label])
                Y.append(label)

            else:
                print("WARNING: shape of a sample is incorrect, not adding it")
                print(f"Error shape is: {sample.shape}")
                print(f"Error vals: {turb_raw[peak_idx - left : peak_idx + right + 1]}")

        assert len(X) == len(Y)

        # save data
        self.data = X
        self.truths = Y

    def __getitem__(self, idx):
        """
        returns the sample and label at index

        RETURNS A SAMPLE WHERE:
        sample[0] = raw data
            sample[0][0] = fdom raw data
            sample[0][1] = stage raw data
            sample[0][2] = turb raw data
            sample[0][3] = time
        sample[1] = int, the label for the data encoded with passed in encoder to init
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data[idx], self.truths[idx])

        return sample

    def __len__(self):
        """
        return the length of the dataset
        """

        return len(self.data)


class turbAugOnlyDataset(data.Dataset):
    """
    augmented only for class balance
    """

    def __init__(
        self,
        labeler,
        fdom_augmented_dir,
        stage_augmented_dir,
        turb_augmented_dir,
        turb_labeled_aug_dir,
        fpt_lookup_filename,
        window_size=15,
    ) -> None:
        super(turbAugOnlyDataset).__init__()

        self.label_encoder = labeler

        # augmented paths
        self.fdom_aug_path = fdom_augmented_dir
        self.stage_aug_path = stage_augmented_dir
        self.turb_aug_path = turb_augmented_dir

        # labeled data
        self.turb_aug_labeled_path = turb_labeled_aug_dir

        self.fpt_lookup_filename = fpt_lookup_filename

        # generate the dataset
        self.get_data()

        print(f"{len(self.data)} candidates found in class-balanced augmented dataset.")

    def get_data(self):
        # use normal timeseries, as we are not cutting out specific data
        # indices select only the second row, which are the respective values
        fdom_raw = np.array(dm.read_in_timeseries(self.fdom_aug_path, True))[:, 1]
        stage_raw = np.array(dm.read_in_timeseries(self.stage_aug_path, True))[:, 1]
        turb_raw = np.array(dm.read_in_timeseries(self.turb_aug_path, True))

        # get the time indices
        time = copy.deepcopy(turb_raw)
        time = time[:, 0]

        # make turb just the values
        turb_raw = turb_raw[:, 1]

        peaks = get_all_cands_turb(self.turb_aug_path, self.turb_aug_labeled_path, True)

        truths = get_all_truths(self.turb_aug_labeled_path)

        # instantiate arrays to load values into to be saved
        X = []
        Y = []

        for i, peak in peaks.iterrows():
            # get start and end indices
            peak_idx = int(peak["idx_of_peak"])
            left_base = int(peak["left_base"])
            right_base = int(peak["right_base"])

            left = abs(peak_idx - left_base)
            right = abs(peak_idx - right_base)

            # each sample follows this order: 0 = fdom, 1 = stage, 2 = turb, 3 = time
            sample = [
                fdom_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                stage_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                turb_raw[peak_idx - left : peak_idx + right + 1].tolist(),
                time[peak_idx - left : peak_idx + right + 1].tolist(),
            ]

            if len(sample) >= 4 and len(sample[0]) > 0:
                X.append(sample)

                # get label
                label = truths.loc[
                    truths["idx_of_peak"] == peak_idx, "label_of_peak"
                ].iloc[0]

                # convert label to normalized integer value, using passed in label encoder
                label = self.label_encoder.transform([label])
                Y.append(label)

            else:
                print("WARNING: shape of a sample is incorrect, not adding it")
                print(f"Error shape is: {sample.shape}")
                print(f"Error vals: {turb_raw[peak_idx - left : peak_idx + right + 1]}")

        assert len(X) == len(Y)

        # save data
        self.data = X
        self.truths = Y

    def __getitem__(self, idx):
        """
        returns the sample and label at index

        RETURNS A SAMPLE WHERE:
        sample[0] = raw data
            sample[0][0] = fdom raw data
            sample[0][1] = stage raw data
            sample[0][2] = turb raw data
            sample[0][3] = time
        sample[1] = int, the label for the data encoded with passed in encoder to init
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data[idx], self.truths[idx])

        return sample

    def __len__(self):
        """
        return the length of the dataset
        """

        return len(self.data)
