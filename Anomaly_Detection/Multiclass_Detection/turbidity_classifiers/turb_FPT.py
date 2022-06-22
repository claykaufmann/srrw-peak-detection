import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import math


class turb_FPT_Classifier:
    """
    class representing a turbidity flat plateau classifier
    """

    def __init__(
        self, turb_data, flatness_range=(0.1, 0.25), prominence_range=(40, 700)
    ) -> None:
        """
        create the FPT classifier
        """
        self.predictions = []

        self.params = {}
        self.best_params = {}

        self.best_acc = 0
        self.best_f1 = 0

        self.flatness_range = flatness_range
        self.prominence_range = prominence_range

        self.turb_data = turb_data

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

    def start_iteration(self):
        """
        call at beginning of iteration
        """
        self.predictions = []
        self.generate_params()

        return self.params

    def generate_params(self):
        """
        generate new params randomly, using random grid search
        """
        params = {}

        params["flatness"] = np.random.uniform(
            self.flatness_range[0], self.flatness_range[1]
        )

        params["prominence"] = np.random.randint(
            self.prominence_range[0], self.prominence_range[1]
        )

        self.params = params

    def classify_samples(self, peaks, use_best_params=False):
        """
        classify samples

        set use_best_params to True for testing
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        results = []

        for peak in peaks:
            left_base = int(peak[1])
            right_base = int(peak[2])
            peak_width = int(right_base - left_base)

            # prominence cond
            prom_cond = peak[3] <= params["prominence"] and peak[3] > 0

            # check flatness
            # flatness defined as min/max val less than 20% apart
            # iterate over peak, and find the min/max val
            min_val = math.inf
            max_val = -math.inf
            for index in range(1, peak_width + 1):
                curr_amp = self.turb_data[left_base + index][1]
                if curr_amp < min_val:
                    min_val = curr_amp

                if curr_amp > max_val:
                    max_val = curr_amp

            # get average value
            avg_val = (min_val + max_val) / 2
            low_bound = avg_val * (1 - params["flatness"])
            high_bound = avg_val * (1 + params["flatness"])

            # check if each value is within 10% of avg val
            if (
                low_bound <= min_val <= high_bound
                and low_bound <= max_val <= high_bound
            ):
                flat_cond = True
            else:
                flat_cond = False

            # check plataeau cond
            # see if one past left base and right base is lower than those values
            plat_cond = True

            # need this try/except in case we get an out of bounds error
            try:
                if self.turb_data[left_base - 2][1] >= self.turb_data[left_base][1]:
                    plat_cond = False
                if self.turb_data[right_base + 2][1] >= self.turb_data[right_base][1]:
                    plat_cond = False
            except:
                plat_cond = False

            # if prom flat and plat conds, this is a flat plateau
            if flat_cond and prom_cond and plat_cond:
                results.append([peak[0], "FPT"])
            else:
                results.append([peak[0], "NAP"])

        self.predictions = results
        return results

    def got_best_results(self):
        """
        save best params
        """
        self.best_params = copy.deepcopy(self.params)

    def end_of_iteration(self, truths):
        """
        test results from past iteration of training
        """
        # check predictions
        TP, TN, FP, FN, results = self.check_predictions(truths)

        # calculate stats
        TPR = 0 if TP == FN == 0 else TP / (TP + FN)
        TNR = TN / (TN + FP)
        bal_acc = (TPR + TNR) / 2
        f1_score = 0 if TP == FP == FN == 0 else (2 * TP) / ((2 * TP) + FP + FN)

        if f1_score > self.best_f1:
            self.best_f1 = f1_score

        acc = bal_acc
        # see if this is the new best
        if acc > self.best_acc:
            # if so, append it
            self.best_acc = acc

    def check_predictions(self, truths):
        """
        check preds for past iteration
        """
        TP = TN = FP = FN = 0
        results = []

        # test classifier
        for i in range(len(self.predictions)):
            pred = self.predictions[i][1]

            truth = truths[i][2]

            if pred == "FPT":
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

        # return information
        return (TP, TN, FP, FN, results)

    def display_results(self):
        """
        display conf matrix
        """
        mean_cfmx = np.zeros((2, 2))
        for key in self.accumulated_cfmxs.keys():
            mean_cfmx += self.accumulated_cfmxs[key]
        mean_cfmx = mean_cfmx / len(self.accumulated_cfmxs)

        print(mean_cfmx)

        plt.figure(figsize=(10, 7))
        plt.title(label="Turbidity Flat Plateau")

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

        plt.figure(figsize=(10, 7))
        plt.title(label="Turbidity Flat Plateau")

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
        plt.xlabel("Ground Truths")
        plt.ylabel("Predictions")
        plt.show()
