import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import confusion_matrix
from Multiclass_Detection.get_cands import get_all_cands_fDOM

sys.path.insert(1, "../../")
from Tools.get_candidates import get_candidates


class fDOM_PLP_Classifier:
    """
    this class represents an fDOM plummeting peak classifier
    """

    def __init__(
        self,
        fdom_data,
        turb_data,
        fdom_filename,
        fdom_truths_filename,
        fdom_augmented_filename=None,
        fdom_augmented_truths_filename=None,
        basewidth_range=(1, 10),
        prominence_range=(5, 20),
        peak_prox_bounds=(1, 20),
        turb_interference_bounds=(0, 10),
    ) -> None:
        """
        creates the fDOM PLP classifier

        PARAMS:
        basewidth_range: the range of which the width needs to be for the peak
        prominence_range: the range of possible prominence values
        peak_prox_bounds: how close another peak can be

        turb_interference_bounds: how close a turb peak can be to classify it is possible interference

        NOTE: values were selected in conjunction with domain scientist
        """

        # init predictions
        self.predictions = []

        # save params dict
        self.params = {}
        self.best_params = {}

        # stats to keep track of best acc and f1 score
        self.best_acc = 0
        self.best_f1_score = 0

        # save values for classifying peaks
        self.basewidth_range = basewidth_range
        self.prominence_range = prominence_range
        self.peak_proximity_bounds = peak_prox_bounds
        self.turb_interference_bounds = turb_interference_bounds

        # create dictionaries for the accumulation stats
        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

        # generate all of the close turb peaks from passed in data
        self.preprocess_turb_interference(
            fdom_data,
            turb_data,
            fdom_filename,
            fdom_augmented_filename,
            fdom_truths_filename,
            fdom_augmented_truths_filename,
        )

    def start_iteration(self):
        """
        call this at the beginning of an iteration
        """
        # empty the predictions list
        self.predictions = []

        # generate new params
        self.generate_params()

        # return params for information (prob not needed)
        return self.params

    def classify_samples(self, peaks, use_best_params=False) -> str:
        """
        classify the given sample as either not anomaly or anomaly

        PARAMETERS:
        index: the index of a peak
        peak: the peak itself
        use_best_params: whether or not to use best params, used for testing at end of iteration

        RETURNS:
        result, 0 if not anomaly, 1 if plummeting peak
        """
        if use_best_params:
            params = self.best_params
        else:
            params = self.params

        results = []
        for i, peak in enumerate(peaks):
            prominence_condition = peak[3] >= params["min_prominence"]

            basewidth_condition = abs(peak[1] - peak[2]) <= params["max_basewidth"]

            interference_condition = (
                self.proximity_to_interference[i, 0]
                >= params["interference_x_proximity"]
                and self.proximity_to_interference[i, 1]
                >= params["interference_y_proximity"]
            )

            proximity_condition = (
                self.proximity_to_adjacent[i] >= params["proximity_threshold"]
            )

            if (
                prominence_condition
                and basewidth_condition
                and interference_condition
                and proximity_condition
            ):
                results.append([peak[0], "PLP"])
            else:
                results.append([peak[0], "NAP"])

        self.predictions = results
        return results

    def generate_params(self):
        """
        generate new params randomly
        """
        # randomly gen params, save them to the class
        params = {}

        params["max_basewidth"] = np.random.randint(
            self.basewidth_range[0], self.basewidth_range[1] + 1
        )
        params["min_prominence"] = np.random.uniform(
            self.prominence_range[0], self.prominence_range[1]
        )

        params["interference_x_proximity"] = np.random.randint(
            self.turb_interference_bounds[0], self.turb_interference_bounds[1]
        )
        params["interference_y_proximity"] = np.random.randint(
            self.turb_interference_bounds[0], self.turb_interference_bounds[1]
        )

        params["proximity_threshold"] = np.random.randint(
            self.peak_proximity_bounds[0], self.peak_proximity_bounds[1]
        )

        self.params = params

    def got_best_result(self):
        """
        main classifier got the best result, we now save our best params
        """
        self.best_params = copy.deepcopy(self.params)

    def end_of_iteration(self, truths):
        """
        test the classifier (used at the end of an iteration)
        """
        # test classifier
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
        check predictions with truths to get statistics of specific classifier
        """
        TP = TN = FP = FN = 0
        results = []

        # test classifier
        for i in range(len(self.predictions)):
            pred = self.predictions[i][1]

            truth = truths[i][2]

            if pred == "PLP":
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

    def preprocess_turb_interference(
        self,
        fDOM,
        turb,
        fdom_filename,
        fdom_augmented_filename,
        fdom_truths_filename,
        fdom_truths_augmented_filename,
    ):
        """
        on init of the classifier, check all close turbidity peaks, to help distinguish between PLP and normal turb interference

        PARAMS:
        fDOM: raw fDOM data
        turb: raw turb data
        """

        # get turbidity peaks
        turb_peak_params = {
            "prom": [6, None],
            "width": [None, None],
            "wlen": 200,
            "dist": 1,
            "rel_h": 0.6,
        }
        turb_peaks, _ = get_candidates(turb, turb_peak_params)

        # get all fdom candidates, and convert them to a single index list to help find turbidity adjacent peaks
        fdom_cands = get_all_cands_fDOM(
            fdom_filename,
            fdom_truths_filename,
        )
        if fdom_augmented_filename != None and fdom_truths_augmented_filename != None:
            fdom_augmented_cands = get_all_cands_fDOM(
                fdom_augmented_filename, fdom_truths_augmented_filename, True
            )
            fdom_cands = pd.concat([fdom_cands, fdom_augmented_cands])
        else:
            print("WARNING: NO AUGMENTED DATA PROVIDED TO PLP CLASSIFIER.")

        del fdom_cands["left_base"]
        del fdom_cands["right_base"]
        del fdom_cands["amplitude"]
        peaks = fdom_cands.squeeze().tolist()

        # find adjacent turbidity peaks
        proximity_to_adjacent = np.zeros((len(peaks)))
        for i in range(len(peaks)):
            x = y = fDOM.shape[0] + 1

            if i > 0:
                x = abs(peaks[i] - peaks[i - 1])
            if i < len(peaks) - 1:
                y = abs(peaks[i] - peaks[i + 1])

            proximity_to_adjacent[i] = min(x, y)

        # find interfering turb peaks
        proximity_to_interference = np.zeros((len(peaks), 2))
        for i, peak in enumerate(peaks):
            x = y = fDOM.shape[0] + 1
            for turb_peak in turb_peaks:
                if turb_peak <= peak:
                    x = min(abs(peak - turb_peak), x)
                else:
                    y = min(abs(peak - turb_peak), y)
            proximity_to_interference[i, 0] = x
            proximity_to_interference[i, 1] = y

        self.proximity_to_adjacent = proximity_to_adjacent
        self.proximity_to_interference = proximity_to_interference

    def display_results(self):
        """
        display results
        """
        mean_cfmx = np.zeros((2, 2))

        for key in self.accumulated_cfmxs.keys():
            mean_cfmx += self.accumulated_cfmxs[key]

        mean_cfmx = mean_cfmx / len(self.accumulated_cfmxs)
        print(mean_cfmx)

        plt.figure(figsize=(10, 7))
        plt.title(label="fDOM Plummeting Peak")

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
        plt.title(label="fDOM Plummeting Peak")

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

    def clasifier_testing(self, split, cands, truths):
        """
        perform end of training tests, and display results
        """
        test_preds = self.test_results(cands)

        TP, TN, FP, FN, results = self.label_test_results(test_preds, truths)

        cfmx = confusion_matrix(
            [row[2] for row in truths],
            [row[1] for row in test_preds],
            labels=["NPLP", "PLP"],
        )

        self.accumulated_cfmxs[split] = copy.deepcopy(cfmx)

        return (TP, TN, FP, FN, results)
