import copy
from Tools.get_all_cands import get_all_cands_fDOM
from Tools.get_candidates import get_candidates
import Tools.data_processing as dp
import numpy as np


class fDOM_FPT_Classifier:
    """
    class represents an fDOM flat plateau classifier
    """

    def __init__(
        self,
        fdom_data,
        fpt_lookup_file_path,
        basewidth_range=(1, 10),
        prominence_range=(20, 300),
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
            peak_width = peak[2] - peak[1]
            basewidth_cond = peak_width >= params["basewidth"]

            # prominence condition
            prom_cond = peak[3] >= params["prominence"]

            # if basewidth and prom, this is a flat plateau
            if basewidth_cond and prom_cond:
                results.append([peak[0], "FPT"])
            else:
                results.append([peak[0], "NFPT"])

        self.predictions = results
        return results

    def got_best_result(self):
        """
        classfier got its best result, save the params
        """
        self.best_params = copy.deepcopy(self.params)

    def preprocess_sample(self):
        """
        preprocess sample as needed
        """
        pass

    def generate_params(self):
        """
        generate new params randomly, using random grid search
        """
        params = {}

        params["basewidth"] = np.random.randint(
            self.basewidth_range[0], self.basewidth_range[1] + 1
        )

        params["prominence"] = np.random.randint(
            self.prominence_range[0], self.prominence_range[1]
        )

    # NOTE: this is prob not needed, implement last
    def check_predictions(self, truths):
        """
        check preds from just completed iteration

        PARAMS:
        truths: a list of all truths for candidate peaks
        """
        pass
