import pandas as pd
import numpy as np
import sys
import copy
from Multiclass_Detection.get_cands import get_all_cands_fDOM
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

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
        augment_data_starting_timestamp,
        x_bounds=(0, 100),
        y_bounds=(0, 100),
        ratio_threshold_range=(0, 20),
    ) -> None:
        """
        creates the classifier

        PARAMS:
        fdom_data: a time-series of fdom data
        stage_data: a time-series of stage data
        augment_data_starting_timestamp: the timestamp where data goes from real to augmented data
        x_bounds: x bounds for proximity stage rises
        y_bounds: y bounds for proximity stage rises
        ratio_threshold_range: the ratio of width to height for a peak
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

        # save augment data beginnning timestamp
        self.augment_begin = augment_data_starting_timestamp

        # save fdom data
        self.fdom_data = fdom_data

        # preprocess stage rises
        self.preprocess_stage_rises(stage_data)

    def start_iteration(self):
        """
        to be called at the start of a training iteration
        clears preds from last iteration, and generates new params
        """

        # empty preds list
        self.predictions = []

        # gen params
        self.generate_params()

        return self.params

    def classify_samples(self, peaks, use_best_params=False) -> str:
        """
        classify the passed in sample

        PARAMS:
        index: index of the peak in list of candidates
        peak: the peak itself
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        # preprocess the sample
        peak = self.preprocess_samples(peaks)

        results = []
        for i, peak in enumerate(peaks):

            # stage rise cond
            stage_rise_cond = not (peak[4] != -1 and peak[4] <= params["x"]) or (
                peak[5] != -1 and peak[5] <= params["y"]
            )

            # peak is not in fall (for non augmented data)
            fall_range_cond = peak[6] == "NFL"

            # prom to basewidth ratio cond
            pbwr = peak[3] / abs(peak[1] - peak[2])
            pbwr_condition = pbwr > params["ratio_threshold"]

            if stage_rise_cond and fall_range_cond and pbwr_condition:
                results.append([peak[0], "PP"])

            else:
                results.append([peak[0], "NAP"])

        self.predictions = results
        return results

    def preprocess_samples(self, peaks):
        """
        add close stage conds, and add the not fall, or fall information to this peak

        PARAMS:
        peak: the candidate peak
        """

        for peak in peaks:

            # stop out of bounds error
            if peak[0] < len(self.s_index) - 1:
                # add close stage conds
                peak.append(self.s_index[int(peak[0]), 0])
                peak.append(self.s_index[int(peak[0]), 1])

            # if it is greater, just assume no stage
            else:
                peak.append(-1)
                peak.append(-1)

            # check if sample is augmented (we can use the timestamp trick)
            # use fdom data to get the actual timestamp
            cand_timestamp = self.fdom_data[int(peak[0]), 0]
            if cand_timestamp > self.augment_begin:
                # the peak is augmented, append not fall, as we can't make month assumptions
                peak.append("NFL")

            else:
                # else, this is from real data, check what month it is coming from
                dt = dp.julian_to_datetime(cand_timestamp)

                if (dt.month == 10) or (dt.month == 9 and dt.day >= 20):
                    peak.append("FL")
                else:
                    peak.append("NFL")

        return peaks

    def generate_params(self):
        """
        generate new params for a new iteration
        """
        params = {}

        # randomly assign new params for random grid search
        params["x"] = np.random.randint(self.x_bounds[0], self.x_bounds[1] + 1)
        params["y"] = np.random.randint(self.y_bounds[0], self.y_bounds[1] + 1)
        params["ratio_threshold"] = np.random.uniform(
            self.ratio_threshold_range[0], self.ratio_threshold_range[1]
        )

        # save these params
        self.params = params

    def got_best_results(self):
        """
        return curr best results
        """
        self.best_params = copy.deepcopy(self.params)

    def end_of_iteration(self, truths):
        """
        test the results from past iteration of training (call at end of iteration)

        PARAMS:
        truths: a list of truths for all candidates
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
        check predictions for the past iteration

        PARAMS:
        truths: a list of truths for all candidates
        """

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

        # return information
        return (TP, TN, FP, FN, results)

    def preprocess_stage_rises(self, stage_data):
        """
        get all stage rises, and add them to an array maintaining close prox to fDOM peaks

        PARAMS:
        stage_data: a time-series of stage data
        """

        # call detect_stage_rises to get stage rises
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

        # save this information for use later
        self.s_index = s_indexed

    def test_results(self, peaks):
        """
        used to test the classifier on test data
        """

        params = self.best_params
        results = []
        for i, peak in enumerate(peaks):
            peak = self.preprocess_sample(peak)

            # Peak is not near stage rise
            stage_rise_condition = not (peak[4] != -1 and peak[4] <= params["x"]) or (
                peak[4] != -1 and peak[5] <= params["y"]
            )
            # Peak is not in fall
            fall_range_condition = peak[6] == "NFL"
            # Peak has a large enough prominence/basewidth ratio
            pbwr = peak[5] / abs(peak[1] - peak[2])
            pbwr_condition = pbwr > params["ratio_threshold"]

            if stage_rise_condition and fall_range_condition and pbwr_condition:
                results.append([peak[0], "PP"])
            else:
                results.append([peak[0], "NPP"])

        self.predictions = results
        return results

    def label_test_results(self, preds, truths):
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
        display test results in a heatmap
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

    def classifier_testing(self, split, cands, truths):
        """
        perform end of split tests, display results
        """
        test_preds = self.test_results(cands)

        TP, TN, FP, FN, results = self.label_test_results(test_preds, truths)

        cfmx = confusion_matrix(
            [row[2] for row in truths],
            [row[1] for row in test_preds],
            labels=["NPP", "PP"],
        )

        self.accumulated_cfmxs[split] = copy.deepcopy(cfmx)

        return (TP, TN, FP, FN, results)
