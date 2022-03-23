import pandas as pd
import numpy as np
import sys
import copy
from Multiclass_Detection.get_cands import get_all_cands_fDOM

sys.path.insert(1, "../../")
from Tools.get_candidates import get_candidates
import Tools.data_processing as dp


class fDOM_FPT_Classifier:
    """
    class represents an fDOM flat plateau classifier
    """

    def __init__(self) -> None:
        """
        creates the flat plateau classifier
        """
        pass

    def start_iteration(self):
        """
        call at beginning of an iteration
        """
        pass

    def classify_sample(self):
        """
        classify the given sample as either not anomaly or anomaly
        """
        pass

    def preprocess_sample(self):
        """
        preprocess sample as needed
        """
        pass

    def generate_params(self):
        """
        generate new params randomly, using random grid search
        """
        pass

    def test_results(self, truths):
        """
        test the results from just completed iteration

        PARAMS:
        truths: a list of all truths for candidate peaks
        """
        pass

    def check_predictions(self, truths):
        """
        check preds from just completed iteration

        PARAMS:
        truths: a list of all truths for candidate peaks
        """
        pass
