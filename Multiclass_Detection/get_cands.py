import sys

sys.path.insert(1, "../")
import Tools.get_candidates as gc
import pandas as pd


# ~~~~~~~ FDOM ~~~~~~~


def get_all_cands_fDOM(
    raw_fdom_data_filename, truths_filename, is_augmented=False
) -> pd.DataFrame:
    """
    get all fdom candidates

    PARAMS:
    raw_fdom_data_filename: raw fdom data filename
    truths_filename: truths filename
    is_augmented: if the passed in data is augmented
    """
    # get candidates from normal data
    cands_skp = gc.get_cands_fDOM_SKP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    cands_plp = gc.get_cands_fDOM_PLP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    # this makes all plp candidate amplitudes negative, as all of these are downward peaks
    # used in PLP detection, where the multiclass classifier checks to make sure the amplitude is negative
    # cands_plp["amplitude"] = cands_plp["amplitude"] * -1

    cands_pp = gc.get_cands_fDOM_PP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    cands_nap = gc.get_cands_fDOM_NAP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    if not is_augmented:
        cands_fpt = gc.get_cands_fDOM_FPT()
        cands_fsk = gc.get_cands_fDOM_FSK()

        # concat dataframes
        all_cands = pd.concat(
            [cands_skp, cands_plp, cands_pp, cands_fpt, cands_fsk, cands_nap]
        )
    else:
        # concat dataframes
        all_cands = pd.concat([cands_skp, cands_plp, cands_pp, cands_nap])

    all_cands = all_cands.sort_values(by=["idx_of_peak"], kind="stable")

    all_cands = all_cands.set_index("idx_of_peak")
    all_cands = all_cands[~all_cands.index.duplicated(keep="first")]

    all_cands = all_cands.reset_index()

    return all_cands


def get_all_truths_fDOM(filename, is_augmented=False) -> pd.DataFrame:
    """
    get all the truths for fdom

    PARAMS:
    filename: path to the fdom truths file
    is_augmented: if the data being passed is augmented (needed as currently no way to get FPT and FSK augmented values into these cand lists)
    """
    # load in the entire truths file
    truths = pd.read_csv(filename)
    return truths


# ~~~~~~~ TURBIDITY ~~~~~~~
