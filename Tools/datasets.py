"""
this file contains the dataset for pytorch learning
"""

import torch
import torch.utils.data as data
import numpy as np
import sys
import pandas as pd

from . import data_movement as dm
from . import data_processing as dp
from .get_all_cands import get_all_cands_fDOM, get_all_truths_fDOM


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

        self.get_data()

    def get_data(self):
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
        cands = get_all_cands_fDOM(self.fdom_raw_path, self.fdom_raw_labeled_path)

        # get all respective truths
        truths = get_all_truths_fDOM(self.fdom_raw_labeled_path)

        # load in augmented data
        if self.fdom_aug_path is not None:
            # use normal timeseries, as we are not cutting out specific data
            fdom_aug = np.array(dm.read_in_timeseries(self.fdom_aug_path, True))
            stage_aug = np.array(dm.read_in_timeseries(self.stage_aug_path, True))
            turb_aug = np.array(dm.read_in_timeseries(self.turb_aug_path, True))

            # labeled aug data
            fdom_aug_labeled = get_all_truths_fDOM(self.fdom_aug_labeled_path, True)

            # get all cands from augmented data, augment cands
            aug_cands = get_all_cands_fDOM(
                self.fdom_raw_path, self.fdom_aug_labeled_path, True
            )

            aug_truths = get_all_truths_fDOM(self.fdom_aug_labeled_path, True)

            # concat these two frames
            cands = pd.concat([cands, aug_cands])
            truths = pd.concat([truths, aug_truths])

        # now, we need to modify these cands so they are saved in the following format:
        # (sample, label), where sample is a 3 x m array, representing a segment of data for each fdom, stage, turb,
        #   and label is the truth label

        # get the starting and ending points for each candidate...

    def __len__(self):
        """
        returns len of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        returns the sample and label at index
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # collect the sample, its a vector of fdom, stage, and turb
        sample = (self.data[idx], self.truths[idx])

        return sample


class turbidityDataset(data.Dataset):
    """
    turbidity dataset
    """
