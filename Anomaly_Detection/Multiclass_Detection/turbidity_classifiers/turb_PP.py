import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from Tools.auxiliary_functions import detect_stage_rises
import Tools.data_processing as dp


class turb_PP_Classifier:
    """
    a turbidity phantom peak classifier
    """

    def __init__(
        self,
        turb_data,
        stage_data,
        augmented_starting_timestamp,
        x_bounds=(0, 30),
        y_bounds=(0, 30),
        ratio_threshold_range=(0, 50),
    ) -> None:
        """
        creates the classifier
        """

        self.preds = []

        self.params = {}
        self.best_params = {}

        self.best_acc = 0
        self.best_f1 = 0

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.ratio_threshold_range = ratio_threshold_range

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        self.augment_begin = augmented_starting_timestamp

        self.turb_data = turb_data
        self.preprocess_stage_rises(stage_data)

    def start_iteration(self):
        """
        start iteration
        """
        self.preds = []

        self.generate_params()

        return self.params

    def classify_samples(self, peaks, use_best_params=False) -> str:
        pass
