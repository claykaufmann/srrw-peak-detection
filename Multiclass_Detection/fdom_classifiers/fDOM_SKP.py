from email.mime import base
import pandas as pd
import numpy as np
import sys
import copy
from Multiclass_Detection.get_cands import get_all_cands_fDOM
from scipy.signal import find_peaks

sys.path.insert(1, "../../")
from Tools.get_candidates import get_candidates
import Tools.data_processing as dp


class fDOM_SKP_Classifier:
    """
    this class represents a skyrocketing peak fDOM classifier
    """

    def __init__(
        self,
        fDOM,
        fdom_raw_filename,
        fdom_truth_filename,
        fdom_augmented_filename=None,
        fdom_augmented_truth_filename=None,
        basewidth_range=(1, 10),
        prominence_range=(5, 20),
        peak_prox_bounds=(0, 20),
        downward_bases_range=(0, 5),
    ) -> None:
        """
        fDOM: fDOM data
        """
        # init predictions
        self.predictions = []

        # init results
        self.results = []

        # save params dict
        self.params = {}
        self.best_params = {}

        # save vals passed in
        self.basewidth_range = basewidth_range
        self.prominence_range = prominence_range
        self.peak_proximity_bounds = peak_prox_bounds
        self.downward_bases_range = downward_bases_range

        # stats to keep track of best acc and f1 score
        self.best_acc = 0
        self.best_f1_score = 0

        # create dictionaries for the accumulation stats
        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        self.initialize_detection_mechanisms(
            fDOM,
            fdom_raw_filename,
            fdom_truth_filename,
            fdom_augmented_filename,
            fdom_augmented_truth_filename,
        )

    def start_iteration(self):
        """
        call this at the beginning of a train iteration, performs cleanup and generates params
        """

        self.predictions = []

        self.generate_params()

        return self.params

    def generate_params(self):
        """
        generates new params
        """

        # Random grid search for hyperparams
        params = {}

        params["max_basewidth"] = np.random.randint(
            self.basewidth_range[0], self.basewidth_range[1] + 1
        )
        params["min_prominence"] = np.random.uniform(
            self.prominence_range[0], self.prominence_range[1]
        )

        params["downward_bases_threshold"] = np.random.randint(
            self.downward_bases_range[0], self.downward_bases_range[1]
        )

        params["proximity_threshold"] = np.random.randint(
            self.peak_proximity_bounds[0], self.peak_proximity_bounds[1]
        )

        self.params = params

    def classify_sample(self, index, peak):
        """
        classify a peak
        """
        prom_cond = peak[3] >= self.params["min_prominence"]
        basewidth_cond = abs(peak[1] - peak[2]) <= self.params["max_basewidth"]
        downward_bases_cond = self.check_downward_peak_condition(index)
        peak_prox_cond = (
            self.prox_to_adjacent[index] >= self.params["proximity_threshold"]
        )

        if prom_cond and basewidth_cond and downward_bases_cond and peak_prox_cond:
            self.predictions.append([peak[0], "SKP"])
            return "SKP"

        else:
            self.predictions.append([peak[0], "NAP"])
            return "NAP"

    def check_downward_peak_condition(self, index):
        """
        check to see if downward peak is close by
        """
        left = (
            self.prox_to_downward[index, 0] <= self.params["downward_bases_threshold"]
        )
        right = (
            self.prox_to_downward[index, 1] <= self.params["downward_bases_threshold"]
        )

        if left and right:
            return False
        if left and not right:
            return True
        return True

    def test_results(self, truths):
        """
        test the results of the past iteration
        """
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
        """
        check all predictions
        """
        TP = TN = FP = FN = 0
        results = []

        for i in range(len(self.predictions)):
            prediction = self.predictions[i][1]
            truth = truths[i][2]

            if prediction == "SKP":
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

    def initialize_detection_mechanisms(
        self,
        fDOM,
        fdom_raw_filename,
        fdom_truth_filename,
        fdom_augmented_filename,
        fdom_augmented_truth_filename,
    ):
        """
        get downward peaks, and prox peaks
        """
        # get total list of candidates
        fdom_cands = get_all_cands_fDOM(fdom_raw_filename, fdom_truth_filename)

        # add augmented peaks
        if fdom_augmented_filename and fdom_augmented_truth_filename != None:
            fdom_aug_cands = get_all_cands_fDOM(
                fdom_augmented_filename, fdom_augmented_truth_filename, True
            )
            fdom_cands = pd.concat([fdom_cands, fdom_aug_cands])

        cands = fdom_cands.values.tolist()

        del fdom_cands["left_base"]
        del fdom_cands["right_base"]
        del fdom_cands["amplitude"]
        peaks = fdom_cands.squeeze().tolist()

        # GET PROXIMITY PEAKS
        prox_to_adjacent = np.zeros((len(peaks)))

        for i in range(len(peaks)):
            x = y = fDOM.shape[0] + 1
            if i > 0:
                x = abs(peaks[i] - peaks[i - 1])
            if i < len(peaks) - 1:
                y = abs(peaks[i] - peaks[i + 1])

            prox_to_adjacent[i] = min(x, y)

        self.prox_to_adjacent = prox_to_adjacent

        # GET DOWNWARD PEAKS
        flipped_fDOM = dp.flip_timeseries(copy.deepcopy(fDOM))

        prominence_range = [3, None]  # peaks must have at least prominence 3
        width_range = [None, 10]  # peaks cannot have a base width of more than 5
        wlen = 100
        distance = 1
        rel_height = 0.6

        # Get list of all peaks that could possibly be plummeting peaks
        downward_peaks, _ = find_peaks(
            flipped_fDOM[:, 1],
            height=(None, None),
            threshold=(None, None),
            distance=distance,
            prominence=prominence_range,
            width=width_range,
            wlen=wlen,
            rel_height=rel_height,
        )

        prox_to_downward = np.zeros((len(peaks), 2))

        for i, cand in enumerate(cands):
            x = y = fDOM.shape[0] + 1

            for downward_peak in downward_peaks:
                if downward_peak <= cand[1]:
                    x = min(abs(cand[1] - downward_peak), x)
                elif downward_peak >= cand[2]:
                    y = min(abs(cand[2] - downward_peak), y)

            prox_to_downward[i, 0] = x
            prox_to_downward[i, 1] = y

        self.prox_to_downward = prox_to_downward
