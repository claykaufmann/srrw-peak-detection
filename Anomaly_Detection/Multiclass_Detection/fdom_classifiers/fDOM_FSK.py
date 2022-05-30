import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.manifold import trustworthiness
from sklearn.metrics import confusion_matrix
import pandas as pd


class fDOM_FSK_Classifier:
    """
    class represents an fDOM flat sink classifier
    """

    def __init__(
        self,
        fdom_data,
        basewidth_range=(1, 10),
        prominence_range=(20, 300),
        prominence_difference_range=(0.01, 2),
    ) -> None:
        """
        creates the flat plateau classifier
        """
        self.predictions = []

        self.params = {}
        self.best_params = {}

        self.best_acc = 0
        self.best_f1_score = 0

        self.basewidth_range = basewidth_range
        self.prominence_range = prominence_range
        self.prom_diff_range = prominence_difference_range

        self.fdom_data = fdom_data

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

    def start_iteration(self):
        """
        call at beginning of an iteration
        """
        self.predictions = []

        self.generate_params()

        return self.params

    def classify_samples(self, peaks, use_best_params=False):
        """
        classify the given sample as either not anomaly or anomaly

        peak shape:
            peak[0]: index
            peak[1]: left base
            peak[2]: right base
            peak[3]: prominence
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        results = []
        for i, peak in enumerate(peaks):
            # basewidth condition
            # get the length of the peak, make sure it is within the range of basewidth
            # peak[2] is right base, peak[1] is left base of peak
            peak_width = int(peak[2] - peak[1])
            basewidth_cond = peak_width >= params["basewidth"]

            # prominence condition MIGHT BE AN ISSUE WITH FLAT SINK!
            prom_cond = peak[3] >= params["prominence"]

            # check flatness
            flat_cond = True
            left_base = int(peak[1])
            prev_height = self.fdom_data[left_base][1]
            # iterate over peak, checking flatness
            for index in range(1, peak_width + 1):
                # compare prev height, make sure it is within a given range
                current_height = self.fdom_data[left_base + index][1]
                if abs(current_height - prev_height) > params["prom_diff"]:
                    flat_cond = False
                    break
                prev_height = current_height

            # check sink cond
            # see if one past left base and right base is higher than those values
            sink_cond = True
            right_base = int(peak[2])
            if self.fdom_data[left_base - 1][1] <= self.fdom_data[left_base][1]:
                sink_cond = False
            if self.fdom_data[right_base + 1][1] <= self.fdom_data[right_base][1]:
                sink_cond = False

            # if basewidth prom flat and plat conds, this is a flat plateau
            if basewidth_cond and prom_cond and flat_cond and sink_cond:
                results.append([peak[0], "FSK"])
            else:
                results.append([peak[0], "NFSK"])

        self.predictions = results
        return results

    def got_best_results(self):
        """
        got best result, save params
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

        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score

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

            if pred == "FSK":
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

    def label_test_resulrs(self, preds, truths):
        """
        label test results
        """
        TP = TN = FP = FN = 0
        results = []

        for i in range(len(preds)):
            prediction = preds[i][1]
            truth = truths[i][2]

            if prediction == "NPP":
                if truth == "NPP":
                    TN += 1
                    results.append(preds[i].append("TN"))
                else:
                    FN += 1
                    results.append(preds[i].append("FN"))
            else:
                if truth == "NPP":
                    FP += 1
                    results.append(preds[i].append("FP"))
                else:
                    TP += 1
                    results.append(preds[i].append("TP"))

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
        plt.title(label="fDOM Phantom Peak")

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
        plt.title(label="fDOM Phantom Peak")

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

    def generate_params(self):
        """
        gen new params
        """
        params = {}

        params["basewidth"] = np.random.randint(
            self.basewidth_range[0], self.basewidth_range[1] + 1
        )

        params["prominence"] = np.random.randint(
            self.prominence_range[0], self.prominence_range[1]
        )

        params["prom_diff"] = np.random.uniform(
            self.prom_diff_range[0], self.prom_diff_range[1]
        )

        self.params = params

    def classifier_testing(self, split, cands, truths):
        """
        test classifier
        """
        test_preds = self.classify_samples(cands, use_best_params=True)

        TP, TN, FP, FN, results = self.label_test_results(test_preds, truths)

        cfmx = confusion_matrix(
            [row[2] for row in truths],
            [row[1] for row in test_preds],
            labels=["NPP", "PP"],
        )

        self.accumulated_cfmxs[split] = copy.deepcopy(cfmx)

        return (TP, TN, FP, FN, results)
