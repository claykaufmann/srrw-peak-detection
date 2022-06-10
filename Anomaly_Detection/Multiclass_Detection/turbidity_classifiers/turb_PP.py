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
        fdom_data,
        stage_data,
        augment_data_starting_timestamp,
        x_bounds=(0, 30),  # TODO: modify these bounds
        y_bounds=(0, 30),
        intf_range_1_bounds=(-10, 2),
        intf_range_2_bounds=(-5, 3),
        intf_range_3_bounds=(-2, 10),
        intf_threshold_bounds=(-2, 2),
        ratio_threshold_range=(0, 50),  # may not be needed
        turb_peak_prom_threshold=50,
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
        self.intf_range_1_bounds = intf_range_1_bounds
        self.intf_range_2_bounds = intf_range_2_bounds
        self.intf_range_3_bounds = intf_range_3_bounds
        self.intf_threshold_bounds = intf_threshold_bounds

        # this may not be needed
        self.ratio_threshold_range = ratio_threshold_range

        self.turb_peak_prom_threshold = turb_peak_prom_threshold

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        self.augment_begin = augment_data_starting_timestamp

        self.turb_data = turb_data
        self.fdom_data = fdom_data
        self.preprocess_stage_rises(stage_data)

    def generate_params(self):
        """
        generate new params for random grid search
        """
        params = {}

        params["x"] = np.random.randint(self.x_bounds[0], self.x_bounds[1] + 1)
        params["y"] = np.random.randint(self.y_bounds[0], self.y_bounds[1] + 1)

        temp = np.random.randint(
            self.intf_range_1_bounds[0], self.intf_range_1_bounds[1]
        )
        params["inft_range_1"] = (
            temp,
            np.random.randint(temp, self.intf_range_1_bounds[1]) + 2,
        )

        temp = np.random.randint(
            self.intf_range_2_bounds[0], self.intf_range_2_bounds[1]
        )
        params["inft_range_2"] = (
            temp,
            np.random.randint(temp, self.intf_range_2_bounds[1]) + 2,
        )

        temp = np.random.randint(
            self.intf_range_3_bounds[0], self.intf_range_3_bounds[1]
        )
        params["inft_range_3"] = (
            temp,
            np.random.randint(temp, self.intf_range_3_bounds[1]) + 2,
        )

        params["intf_t1"] = np.random.uniform(
            self.intf_threshold_bounds[0], self.intf_threshold_bounds[1]
        )
        params["intf_t2"] = np.random.uniform(
            self.intf_threshold_bounds[0], self.intf_threshold_bounds[1]
        )

        self.params = params

    def start_iteration(self):
        """
        start iteration
        """
        self.preds = []

        self.generate_params()

        return self.params

    def check_for_fDOM_interference(fdom_data, peak_idx, intf_params):
        """
        given index of turb peak, check fDOM for interference
        """
        # Calculate mean tangent line across range_1
        mean_tan_1 = np.mean(
            np.diff(
                fdom_data[
                    (peak_idx + intf_params["inft_range_1"][0]) : (
                        peak_idx + intf_params["inft_range_1"][1]
                    ),
                    1,
                ]
            )
        )
        # Calculate mean tangent line across range_2
        mean_tan_2 = np.mean(
            np.diff(
                fdom_data[
                    (peak_idx + intf_params["inft_range_2"][0]) : (
                        peak_idx + intf_params["inft_range_2"][1]
                    ),
                    1,
                ]
            )
        )
        # Calculate mean tangent line across range_3
        mean_tan_3 = np.mean(
            np.diff(
                fdom_data[
                    (peak_idx + intf_params["inft_range_3"][0]) : (
                        peak_idx + intf_params["inft_range_3"][1]
                    ),
                    1,
                ]
            )
        )
        # Compare means
        if (mean_tan_1 * intf_params["intf_t1"]) > (mean_tan_2) or (mean_tan_2) < (
            mean_tan_3 * intf_params["intf_t2"]
        ):
            return True
        return False

    def classify_samples(self, peaks, use_best_params=False) -> str:
        """
        classify the samples
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        peaks = self.preprocess_samples(peaks)

        results = []
        for peak in peaks:
            if (peak[4] != -1 and peak[4] <= params["x"]) or (
                peak[5] != -1 and peak[5] <= params["y"]
            ):
                if peak[3] > self.turb_peak_prom_threshold:
                    intf_result = self.check_for_fDOM_interference(
                        self.fdom_data, peak[0], params
                    )

                    if intf_result:
                        results.append([peak[0], "NAP"])
                    else:
                        results.append([peak[0], "PP"])

                else:
                    results.append([peak[0], "NPP"])

            else:
                results.append([peak[0], "PP"])

        self.preds = results
        return results

    def preprocess_samples(self, peaks):
        """
        add close stage conds to peaks
        """

        for peak in peaks:
            peak.append(self.s_index[int(peak[0]), 0])
            peak.append(self.s_index[int(peak[0]), 1])

        return peaks

    def got_best_results(self):
        """
        save params
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

    def preprocess_stage_rises(self, stage_data):
        """
        preprocess stage rises in data
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
        plt.title(label="Turbidty Phantom Peak Ratio")

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
        plt.title(label="Turbidty Phantom Peak Totals")

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
