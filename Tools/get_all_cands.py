import Tools.get_candidates as gc
import pandas as pd


# ~~~~~~~ FDOM/TURB ~~~~~~~
def get_all_truths(filename) -> pd.DataFrame:
    """
    returns all truths in a dataframe

    PARAMS:
    filename: path to the labeled truths file
    """

    truths = pd.read_csv(filename)
    return truths


# ~~~~~~~ FDOM ~~~~~~~
def get_all_cands_fDOM(
    raw_fdom_data_filename,
    truths_filename,
    is_augmented=False,
    fpt_lookup_filename=None,
    fsk_lookup_filename=None,
) -> pd.DataFrame:
    """
    get all fdom candidates

    PARAMS:
    raw_fdom_data_filename: raw fdom data filename
    truths_filename: truths filename
    is_augmented: if the passed in data is augmented
    """
    # NOTE: Because of a major refactor, we need these default args for fpt/fsk lookup table names
    if is_augmented and fpt_lookup_filename == None and fsk_lookup_filename == None:
        # if this is augmented data, and no fpt/fsk passed in, assume top level basic file names
        fpt_lookup_filename = "Data/augmented_data/fdom/fpt_lookup.csv"
        fsk_lookup_filename = "Data/augmented_data/fdom/fsk_lookup.csv"

    # get candidates from normal data
    cands_skp = gc.get_cands_fDOM_SKP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    cands_plp = gc.get_cands_fDOM_PLP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    cands_pp = gc.get_cands_fDOM_PP(
        raw_fdom_data_filename, truths_filename, is_augmented
    )

    cands_nap = gc.get_cands_fDOM_NAP(
        raw_fdom_data_filename,
        truths_filename,
        is_augmented,
        fpt_lookup_filename,
        fsk_lookup_filename,
    )

    cands_fpt = gc.get_cands_fDOM_FPT(
        raw_fdom_data_filename, truths_filename, is_augmented, fpt_lookup_filename
    )
    cands_fsk = gc.get_cands_fDOM_FSK(
        raw_fdom_data_filename, truths_filename, is_augmented, fsk_lookup_filename
    )

    # concat dataframes
    all_cands = pd.concat(
        [cands_skp, cands_plp, cands_pp, cands_fpt, cands_fsk, cands_nap]
    )

    all_cands = all_cands.sort_values(by=["idx_of_peak"], kind="stable")

    all_cands = all_cands.set_index("idx_of_peak")
    all_cands = all_cands[~all_cands.index.duplicated(keep="first")]

    all_cands = all_cands.reset_index()

    return all_cands


def get_all_truths_fDOM(filename, is_augmented=False) -> pd.DataFrame:
    """
    get all the truths for fdom

    NOTE: This function is NOT NEEDED, can use the above get_all_truths function instead

    PARAMS:
    filename: path to the fdom truths file
    is_augmented: if the data being passed is augmented
    """
    # load in the entire truths file
    truths = pd.read_csv(filename)
    return truths


# ~~~~~~~ TURBIDITY ~~~~~~~
def get_all_cands_turb(
    raw_filename, truths_filename, is_augmented=False
) -> pd.DataFrame:
    """
    get all turb cands

    PARAMS:
    raw_filename: path to a turbidity file with raw data
    truths_filename: the path to the labeled data
    is_augmented: is the data augmented or not
    """

    cands_skp = gc.get_cands_turb_SKP(raw_filename, truths_filename, is_augmented)

    cands_pp = gc.get_cands_turb_PP(raw_filename, truths_filename, is_augmented)

    cands_NAP = gc.get_cands_turb_NAP(raw_filename, truths_filename, is_augmented)

    if not is_augmented:
        cands_fpt = gc.get_cands_turb_FPT(raw_filename, truths_filename, is_augmented)

        all_cands = pd.concat([cands_skp, cands_pp, cands_NAP, cands_fpt])
    else:
        all_cands = pd.concat([cands_skp, cands_pp, cands_NAP])

    all_cands = all_cands.sort_values(by=["idx_of_peak"], kind="stable")

    all_cands = all_cands.set_index("idx_of_peak")
    all_cands = all_cands[~all_cands.index.duplicated(keep="first")]

    all_cands = all_cands.reset_index()

    return all_cands
