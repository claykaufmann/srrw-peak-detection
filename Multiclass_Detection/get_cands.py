import sys

sys.path.insert(1, "../")

import Tools.get_candidates as gc
import pandas as pd


def get_all_cands_fDOM() -> pd.DataFrame:
    """
    get all fdom candidates

    TODO: modify function to take in filenames
    """
    # get candidates from normal data
    cands_skp = gc.get_cands_fDOM_SKP()
    cands_plp = gc.get_cands_fDOM_PLP()
    cands_pp = gc.get_cands_fDOM_PP()
    cands_nap = gc.get_cands_fDOM_NAP()
    cands_fpt = gc.get_cands_fDOM_FPT()
    cands_fsk = gc.get_cands_fDOM_FSK()

    # TODO: add functionality to get augmented data cands

    # concat dataframes
    all_cands = pd.concat(
        [cands_skp, cands_plp, cands_pp, cands_fpt, cands_fsk, cands_nap]
    )

    all_cands = all_cands.sort_values(by=["idx_of_peak"], kind="stable")

    all_cands = all_cands.set_index("idx_of_peak")
    all_cands = all_cands[~all_cands.index.duplicated(keep="first")]

    return all_cands


def get_all_truths_fDOM() -> pd.DataFrame:
    """
    get all the truths for fdom

    TODO: modify function to take in filenames
    """
    # load in the entire truths file
    truths_original_data = pd.read_csv(
        "../Data/labeled_data/ground_truths/fDOM/fDOM_all_julian_0k-300k.csv"
    )

    # TODO: read in augmented data here

    # concat data here
    truths = truths_original_data

    truths = truths.set_index("timestamp_of_peak")

    return truths
