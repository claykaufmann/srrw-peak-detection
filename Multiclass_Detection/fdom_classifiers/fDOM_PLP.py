from hashlib import new
from sqlite3 import paramstyle
import pandas as pd
import numpy as np


class fDOM_PLP_Classifier:
    """
    this class represents an fDOM plummeting peak classifier
    """

    def __init__(
        self,
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

        self.accumulated_test_metrics = {}
        self.accumulated_test_results = {}
        self.accumulated_cfmxs = {}

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

    def classify_sample(self, fdom, stage, turb) -> int:
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
        prominence_cond = fdom[3] >= self.params["min_prominence"]
        basewdith_cond = abs(fdom[1] - fdom[2]) <= self.params["max_basewidth"]

        # make prediction
        # TODO: implement this
        pred = []

        # append result to predictions
        self.predictions.append(pred)

    def preprocess_sample(self):
        """
        preprocess the sample data to align with stage, check turbidity interference, etc.
        """
        # find close turb peaks, save them
        proximity_to_adjacent = np.zero

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
