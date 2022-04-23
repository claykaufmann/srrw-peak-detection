from email.mime import base
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

    def __init__(self, basewidth_range=(1, 10), prominence_range=(20, 300)) -> None:
        """
        creates the flat plateau classifier
        """
        self.params = {}
        self.best_params = {}

        self.best_acc = 0

        self.basewidth_range = basewidth_range
        self.prominence_range = prominence_range

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
        pass

    # NOTE: this is prob not needed, implement last
    def check_predictions(self, truths):
        """
        check preds from just completed iteration

        PARAMS:
        truths: a list of all truths for candidate peaks
        """
        pass
