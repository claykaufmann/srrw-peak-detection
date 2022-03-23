import pandas as pd
import numpy as np
import sys
import copy
from Multiclass_Detection.get_cands import get_all_cands_fDOM

sys.path.insert(1, "../../")
from Tools.get_candidates import get_candidates
from Tools.auxiliary_functions import detect_stage_rises
import Tools.data_processing as dp


class fDOM_PP_Classifier:
    """
    represents a phantom peak fDOM classifier
    """

    def __init__(
        self,
        fdom_data,
        stage_data,
        x_bounds=(0, 100),
        y_bounds=(0, 100),
        ratio_threshold_range=(0, 20),
    ) -> None:
        """
        creates the classifier
        """

        # init preds
        self.predictions = []

        # save params dict
        self.params = {}
        self.best_params = {}

        # stats to keep track of acc and f1
        self.best_acc = 0
        self.best_f1_score = 0

        # save vals for classifying peaks
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.ratio_threshold_range = ratio_threshold_range

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        # TODO: perform preprocessing

    def start_iteration(self):
        # empty preds list
        self.predictions = []

        # gen params
        self.generate_params()

        return self.params

    def classify_sample(self, index, peak, augment_timestamp_divide) -> str:
        """
        classify the passed in sample

        PARAMS:
        index: index of the peak in list of candidates
        peak: the peak itself
        augment_timestamp_divide: the timestamp at which augmented data begins (needed for preprocessing)
        """

        # preprocess the sample
        peak = self.preprocess_sample(peak, augment_timestamp_divide)

        # stage rise cond
        stage_rise_cond = not (peak[4] != -1 and peak[4] <= self.params["x"]) or (
            peak[4] != -1 and peak[5] <= self.params["y"]
        )

        # peak is not in fall (for non augmented data)
        # TODO: WATCH INDEX, IT INCLUDES STAGE INFORMATION
        fall_range_cond = peak[6] == "NFL"

        # prom to basewidth ratio cond
        pbwr = peak[3] / abs(peak[1] - peak[2])
        pbwr_condition = pbwr > self.params["ratio_threshold"]

        if stage_rise_cond and fall_range_cond and pbwr_condition:
            self.predictions.append([peak[0], "PP"])

            return "PP"

        else:
            self.predictions.append([peak[0], "NAP"])

            return "NAP"

    def preprocess_sample(self, peak, augment_timestamp_divide):
        """
        add close stage conds, and add the not fall, or fall information to this peak
        """
        # add close stage conds
        peak.append(self.s_index[peak, 0])
        peak.append(self.s_index[peak, 1])

        # check if sample is augmented (we can use the timestamp trick)
        if peak[0] > augment_timestamp_divide:
            # the peak is augmented, append not fall, as we can't make month assumptions
            peak.append("NFL")

        else:

            dt = dp.julian_to_datetime(peak[0])

            if (dt.month == 10) or (dt.month == 9 and dt.day >= 20):
                peak.append("FL")
            else:
                peak.append("NFL")

        return peak

    def generate_params(self):
        params = {}
        params["x"] = np.random.randint(self.x_bounds[0], self.x_bounds[1] + 1)
        params["y"] = np.random.randint(self.y_bounds[0], self.y_bounds[1] + 1)
        params["ratio_threshold"] = np.random.uniform(
            self.ratio_threshold_range[0], self.ratio_threshold_range[1]
        )

        self.params = params

    def test_results(self, truths):
        TP, TN, FP, FN, results = self.check_predictions(truths)

        # calculate stats
        TPR = 0 if TP == FN == 0 else TP / (TP + FN)
        TNR = TN / (TN + FP)
        bal_acc = (TPR + TNR) / 2
        f1_score = 0 if TP == FP == FN == 0 else (2 * TP) / ((2 * TP) + FP + FN)

        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score

        acc = bal_acc
        # see if this is the new best
        if acc > self.best_acc:
            # if so, append it
            self.best_acc = acc
            max_result = copy.deepcopy(results)
            self.best_params = copy.deepcopy(self.params)

    def check_predictions(self, truths):
        TP = TN = FP = FN = 0
        results = []

        # test classifier
        for i in range(len(self.predictions)):
            pred = self.predictions[i][1]

            truth = truths[i][2]

            if pred == "PP":
                if truth == "NAP":
                    FP += 1
                    results.append(self.predictions[i].append("FP"))
                else:
                    TP += 1
                    results.append(self.predictions[i].append("TP"))

            else:
                if truth == "NAP":
                    TN += 1
                    results.append(self.predictions[i].append("TN"))
                else:
                    FN += 1
                    results.append(self.predictions[i].append("FN"))

        return (TP, TN, FP, FN, results)

    def preprocess_stage_rises(self, stage_data):
        """
        get all stage rises, and add them to an array maintaining close prox to fDOM peaks
        """

        s_indices = detect_stage_rises(stage_data[:, 1])

        y = s_indices.shape[0] - 1
        s_indexed = np.zeros((s_indices.shape[0], 2))
        x_count = -1
        y_count = -1

        for x in range(s_indexed.shape[0]):
            if x_count == -1 and s_indices[x] == -1:
                x_count = 0
            if x_count != -1:
                if s_indices[x] == 1:
                    x_count = 0
                    s_indexed[x, 0] = x_count
                else:
                    x_count += 1
                    s_indexed[x, 0] = x_count
            else:
                s_indexed[x, 0] = -1

            # Y Block
            if y_count == -1 and s_indices[y] == 1:
                y_count = 0
            if y_count != -1:
                if s_indices[y] == 1:
                    y_count = 0
                    s_indexed[y, 1] = y_count
                else:
                    y_count += 1
                    s_indexed[y, 1] = y_count
            else:
                s_indexed[y, 1] = -1

            y -= 1

        self.s_index = s_indexed
