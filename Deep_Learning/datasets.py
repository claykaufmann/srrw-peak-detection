"""
this file contains the datasets for the deep learning section of the project
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd


from Tools import augmentation_helpers as augment
from Tools import data_movement as dm
from Tools import data_processing as dp
from Tools.get_all_cands import get_all_cands_fDOM, get_all_truths_fDOM


class fdomDataset(data.Dataset):
    """
    the fdom dataset
    """

    def __init__(
        self,
        fdom_data_raw_dir,
        stage_raw_dir,
        turb_raw_dir,
        fdom_labeled_raw_dir,
        fdom_augmented_dir=None,
        stage_augmented_dir=None,
        turb_augmented_dir=None,
        fdom_labeled_aug_dir=None,
        window_size=15,
    ) -> None:
        """
        constructor

        PARAMS:
        fdom_data_raw_dir: path to raw fdom data
        stage_raw_dir: path to raw stage data
        turb_raw_dir: path to raw turb data
        fdom_augmented_dir: path to augmented fdom data
        stage_augmented_dir: path to augmented stage data (from fdom)
        turb_augmented_dir: path to augmented turb data (from fdom)
        window_size: the width of data to use for each sample, note this is the distance before and after the main peak index
        """

        super(fdomDataset, self).__init__()

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

        # generate the dataset
        self.get_data(window_size)

    def get_data(self, window_size):
        """
        saves the actual data into the classifier
        """

        # load all datasets
        # load non augmented data
        # raw data
        # TODO: this prob isnt needed
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
                self.fdom_raw_path, self.fdom_aug_labeled_path, True
            )

            # get all truths
            aug_truths = get_all_truths_fDOM(self.fdom_aug_labeled_path, True)

            # concat these two frames
            peaks = pd.concat([peaks, aug_peaks])
            truths = pd.concat([truths, aug_truths])

        # initiate arrays for samples and labels that we will read data into
        X = []
        y = []

        for i, peak in peaks.iterrows():
            # get start and end indices
            peak_idx = peaks.loc[i, "idx_of_peak"]
            start_idx = peaks.loc[i, "left_base"]
            end_idx = peaks.loc[i, "right_base"]

            # these indices in the raw fdom timestamp is all we need
            start_timestamp = fdom_raw[start_idx][0]
            end_timestamp = fdom_raw[end_idx][0]

            # use these indices to collect the data for stage and turb
            # TODO: instead of using window size, use the actual timestamps if variable length data is allowed
            # each sample follows this order: 0 = fdom, 1 = stage, 2 = turb
            sample = np.hstack(
                (
                    fdom_raw[peak_idx - window_size : peak_idx + window_size + 1],
                    stage_raw[peak_idx - window_size : peak_idx + window_size + 1],
                    turb_raw[peak_idx - window_size : peak_idx + window_size + 1],
                )
            ).T
            X.append(sample.T)

            # get label
            label = truths.loc[truths["idx_of_peak"] == peak_idx, "label_of_peak"].iloc[
                0
            ]
            y.append(label)

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

        SAMPLE FORMAT:
        index 0: fdom data for peak range within the given window size on dataset initialization

        RETURNS A SAMPLE WHERE:
        sample[0] = raw data
            sample[0][0] = fdom raw data
            sample[0][1] = stage raw data
            sample[0][2] = turb raw data
        sample[1] = string, the label for the data
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
