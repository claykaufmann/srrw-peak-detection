from email.mime import base
import pandas as pd
import numpy as np
import sys
import copy
from Multiclass_Detection.get_cands import get_all_cands_fDOM
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

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

    def classify_samples(self, peaks, use_best_params=False):
        """
        classify a peak
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        results = []
        for index, peak in enumerate(peaks):
            prom_cond = peak[3] >= params["min_prominence"]
            basewidth_cond = abs(peak[1] - peak[2]) <= params["max_basewidth"]
            downward_bases_cond = self.check_downward_peak_condition(
                index, use_best_params
            )
            peak_prox_cond = (
                self.prox_to_adjacent[index] >= params["proximity_threshold"]
            )

            if prom_cond and basewidth_cond and downward_bases_cond and peak_prox_cond:
                results.append([peak[0], "SKP"])

            else:
                results.append([peak[0], "NAP"])

        self.predictions = results
        return results

    def check_downward_peak_condition(self, index, use_best_params=False):
        """
        check to see if downward peak is close by
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        left = self.prox_to_downward[index, 0] <= params["downward_bases_threshold"]
        right = self.prox_to_downward[index, 1] <= params["downward_bases_threshold"]

        if left and right:
            return False
        if left and not right:
            return True
        return True

    def got_best_result(self):
        """
        main classifier got the best result, we now save our best params
        """
        self.best_params = copy.deepcopy(self.params)

    def end_of_iteration(self, truths):
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
        if fdom_augmented_filename != None and fdom_augmented_truth_filename != None:
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

    def test_results(self, peaks):
        """
        used at end of split to test results
        """
        params = self.best_params

        results = []
        for i, peak in enumerate(peaks):

            prominence_condition = peak[3] >= params["min_prominence"]
            basewidth_condition = abs(peak[1] - peak[2]) <= params["max_basewidth"]
            downward_bases_condition = self.check_downward_peak_condition(i)
            peak_proximity_condition = (
                self.prox_to_adjacent[i] >= params["proximity_threshold"]
            )

            if (
                prominence_condition
                and basewidth_condition
                and downward_bases_condition
                and peak_proximity_condition
            ):
                results.append([peak[0], "SKP"])
            else:
                results.append([peak[0], "NSKP"])
        return results

    def label_test_results(self, preds, truths):
        """
        check the results of test
        """
        TP = TN = FP = FN = 0
        results = []

        for i in range(len(preds)):
            prediction = preds[i][1]
            truth = truths[i][2]

            if prediction == "SKP":
                if truth == "NSKP":
                    FP += 1
                    results.append(preds[i].append("FP"))
                else:
                    TP += 1
                    results.append(preds[i].append("TP"))
            else:
                if truth == "NSKP":
                    TN += 1
                    results.append(preds[i].append("TN"))
                else:
                    FN += 1
                    results.append(preds[i].append("FN"))

        return (TP, TN, FP, FN, results)

    def display_results(self):
        """
        display results of classifier
        """

        # Create and display confusion matrices
        mean_cfmx = np.zeros((2, 2))
        for key in self.accumulated_cfmxs.keys():
            mean_cfmx += self.accumulated_cfmxs[key]
        mean_cfmx = mean_cfmx / len(self.accumulated_cfmxs)

        plt.figure(figsize=(10, 7))
        plt.title(label="fDOM Skyrocketing Peaks")
        sn.set(font_scale=1.5)
        sn.heatmap(
            pd.DataFrame(
                mean_cfmx,
                index=["Negative", "Positive"],
                columns=["Negative", "Positive"],
            ),
            annot=True,
            annot_kws={"size": 16},
        )

        plt.show()

        plt.figure(figsize=(10, 7))
        plt.title(label="fDOM Skyrocketing Peaks")

        sn.set(font_scale=1.5)
        sn.heatmap(
            pd.DataFrame(
                mean_cfmx.astype("float") / mean_cfmx.sum(axis=1)[:, np.newaxis],
                index=["Negative", "Positive"],
                columns=["Negative", "Positive"],
            ),
            annot=True,
            annot_kws={"size": 16},
        )
        plt.xlabel("Ground Truths")
        plt.ylabel("Predictions")
        plt.show()

    def classifier_testing(self, split, cands, truths):
        """
        perform end of straining tests, display results
        """
        test_preds = self.test_results(cands)

        TP, TN, FP, FN, results = self.label_test_results(test_preds, truths)

        cfmx = confusion_matrix(
            [row[2] for row in truths],
            [row[1] for row in test_preds],
            labels=["NSKP", "SKP"],
        )

        self.accumulated_cfmxs[split] = copy.deepcopy(cfmx)

        return (TP, TN, FP, FN, results)
