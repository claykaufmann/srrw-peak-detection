import pandas as pd
import copy
import numpy as np
from scipy.signal import find_peaks
import sys

sys.path.insert(1, "../../")
from Tools.get_candidates import get_candidates
import Tools.data_processing as dp


class fDOM_PLP_Classifier:
    """
    this class represents an fDOM plummeting peak classifier
    """

    def __init__(
        self,
        fdom_data,
        turb_data,
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

        TODO: make sure this is right
        peak_prox_bounds: how close another peak can be

        turb_interference_bounds: how close a turb peak can be to classify it is possible interference

        NOTE: values were selected in conjunction with domain scientist
        """

        # init predictions
        self.predictions = []

        # save params dict
        self.params = {}
        self.best_params = {}

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
        self.preprocess_turb_interference(fdom_data, turb_data)

    def start_iteration(self):
        """
        call this at the beginning of an iteration
        """
        # empty the predictions list
        self.predictions = []

        # generate new params
        self.generate_params()

        # for fDOM PLP, generate

        # return params for information (prob not needed)
        return self.params

    def classify_sample(self, index, peak) -> int:
        """
        classify the given sample as either not anomaly or anomaly

        PARAMETERS:
        fdom: a dataframe containing the samples data for fdom
        stage: a dataframe containing the samples data for stage
        turb: a dataframe containing the samples data for turbidity

        RETURNS:
        result, 0 if not anomaly, 1 if plummeting peak
        """
        # use the current params
        prominence_cond = peak[3] >= self.params["min_prominence"]
        basewdith_cond = abs(peak[1] - peak[2]) <= self.params["max_basewidth"]

        # interference_cond =

        # make necessary preprocesses
        self.preprocess_sample()

        # make prediction
        # TODO: implement this
        pred = []

        # append result to predictions
        self.predictions.append(pred)

    def preprocess_sample(self):
        """
        preprocess the sample data to align with stage, check turbidity interference, etc.
        """
        # TODO: implement this function
        pass

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

    def test_results(self, preds, truths):
        """
        test the classifier (used at the end of an iteration)
        """
        # test classifier

        # check if better than previous best

        # if so, update best params and acc metrics

        # append to the acculmated test metrics and params for information post testing
        pass

    def preprocess_turb_interference(self, fDOM, turb):
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

        # get fDOM peaks
        flipped_fDOM = dp.flip_timeseries(copy.deepcopy(fDOM))

        # Get fDOM plummeting peak candidate set using scipy find_peaks()
        prominence_range = [3, None]  # peaks must have at least prominence 3
        width_range = [None, 10]  # peaks cannot have a base width of more than 5
        wlen = 100
        distance = 1
        rel_height = 0.6

        # Get list of all peaks that could possibly be plummeting peaks
        peaks, props = find_peaks(
            flipped_fDOM[:, 1],
            height=(None, None),
            threshold=(None, None),
            distance=distance,
            prominence=prominence_range,
            width=width_range,
            wlen=wlen,
            rel_height=rel_height,
        )

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
