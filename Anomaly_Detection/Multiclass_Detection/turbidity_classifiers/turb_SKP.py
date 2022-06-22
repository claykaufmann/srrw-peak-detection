import pandas as pd
import numpy as np
import copy
from Tools.get_all_cands import get_all_cands_turb
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sn
import Tools.data_processing as dp


class turb_SKP_Classifier:
    """
    represents a skyrocketing peak turbidity classifier
    """

    def __init__(
        self,
        turb_data,
        turb_raw_filename,
        turb_truths_filename,
        turb_augmented_filename=None,
        turb_augmented_truth_filename=None,
        augmented_only=False,
        basewidth_range=(1, 10),
        prominence_range=(5, 20),
        peak_prox_bounds=(0, 20),
        downward_bases_range=(0, 5),
    ) -> None:
        """
        constructor:

        turb_data: raw turb data
        turb_raw_filename: path to raw turb file
        turb_truths_filename: path to turb truths
        turb_augmented_filename: path to raw augmented turb file
        turb_augmented_truth_filename: path to augmented turb truths
        augmented_only: if only using augmented data, set to True
        the rest are parameter ranges used in detection methods
        """
        self.predictions = []

        self.results = []

        self.params = {}
        self.best_params = {}

        self.basewidth_range = basewidth_range
        self.prominence_range = prominence_range
        self.peak_proximity_bounds = peak_prox_bounds
        self.downward_bases_range = downward_bases_range

        self.best_acc = 0
        self.best_f1_score = 0

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        # init detection mechanisms
        self.initialize_detection_mechanisms(
            turb_data,
            turb_raw_filename,
            turb_truths_filename,
            turb_augmented_filename,
            turb_augmented_truth_filename,
            augmented_only,
        )

    def start_iteration(self):
        """
        call at beginning of train iteration
        """
        self.predictions = []

        self.generate_params()

        return self.params

    def generate_params(self):
        """
        gen new params
        """
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
        classify a set of peaks
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
        check to see if a downward peak is close by
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

    def got_best_results(self):
        """
        main classifier got best result, save best params for this classifier
        """
        self.best_params = copy.deepcopy(self.params)

    def end_of_iteration(self, truths):
        """
        test results of past iteration
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
        turb_data,
        turb_raw_filename,
        turb_truths_filename,
        turb_augmented_filename,
        turb_augmented_truth_filename,
        augmented_only,
    ):
        """
        get downard peaks, and prox peaks
        """
        # get all cands
        # if augmented only, only use augmented candidates
        if augmented_only:
            turb_cands = get_all_cands_turb(
                turb_augmented_filename, turb_augmented_truth_filename, True
            )

        # else, we need to get both augmented and non augmented candidates
        else:
            turb_cands = get_all_cands_turb(turb_raw_filename, turb_truths_filename)

            # if we have augmented candidates, load them
            if turb_augmented_filename and turb_augmented_truth_filename:
                turb_aug_cands = get_all_cands_turb(
                    turb_augmented_filename, turb_augmented_truth_filename, True
                )
                turb_cands = pd.concat([turb_cands, turb_aug_cands])

        # convert cands to lists
        cands = turb_cands.values.tolist()

        del turb_cands["left_base"]
        del turb_cands["right_base"]
        del turb_cands["amplitude"]
        peaks = turb_cands.squeeze().tolist()

        # get proximity peaks
        prox_to_adjacent = np.zeros((len(peaks)))

        for i in range(len(peaks)):
            x = y = turb_data.shape[0] + 1
            if i > 0:
                x = abs(peaks[i] - peaks[i - 1])
            if i < len(peaks) - 1:
                y = abs(peaks[i] - peaks[i + 1])

            prox_to_adjacent[i] = min(x, y)

        self.prox_to_adjacent = prox_to_adjacent

        # get downward peaks
        flipped_turb = dp.flip_timeseries(copy.deepcopy(turb_data))

        prominence_range = [3, None]
        width_range = [None, 10]
        wlen = 100
        distance = 1
        rel_height = 0.6

        downward_peaks, _ = find_peaks(
            flipped_turb[:, 1],
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
            x = y = turb_data.shape[0] + 1

            for downward_peak in downward_peaks:
                if downward_peak <= cand[1]:
                    x = min(abs(cand[1] - downward_peak), x)
                elif downward_peak >= cand[2]:
                    y = min(abs(cand[2] - downward_peak), y)

            prox_to_downward[i, 0] = x
            prox_to_downward[i, 1] = y

        self.prox_to_downward = prox_to_downward

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
        plt.title(label="Turbidity Skyrocketing Peaks")
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
        plt.title(label="Turbidity Skyrocketing Peaks")

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
