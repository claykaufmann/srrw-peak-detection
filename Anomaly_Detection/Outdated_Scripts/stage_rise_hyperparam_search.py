# Import libraries and data
import scipy.io as sio
import math
import pickle
import copy
import numpy as np
from scipy.signal import find_peaks
from os.path import dirname, join as pjoin
import datetime
import csv
import Tools.data_processing as dp
import Tools.data_movement as dm
from Tools.auxiliary_functions import get_stage_events, detect_edges

fDOM_raw_data = np.array(
    dm.read_in_timeseries(
        "Data/converted_data/julian_format/fDOM_raw_10.1.2011-9.4.2020.csv", True
    )
)
stage_data = np.array(
    dm.read_in_timeseries(
        "Data/converted_data/julian_format/stage_10.1.11-1.1.19.csv", True
    )
)
turb_data = np.array(
    dm.read_in_timeseries(
        "Data/converted_data/julian_format/turbidity_raw_10.1.2011_9.4.2020.csv", True
    )
)

# Clean and trim data
start_time = 2456042.5  # Beginning of normal data: April 25th 2012
end_time = stage_data[-1, 0]

# Trim timeseries to all start and end on same date
turb_data = dp.trim_timeseries(turb_data, start_time, end_time)
fDOM_raw_data = dp.trim_timeseries(fDOM_raw_data, start_time, end_time)
stage_data = dp.trim_timeseries(stage_data, start_time, end_time)

stage_data = dp.delete_from_timeseries(stage_data, "Data/Misc/delete_date_ranges.txt")
fDOM_raw_data = dp.delete_from_timeseries(
    fDOM_raw_data, "Data/Misc/delete_date_ranges.txt"
)
turb_data = dp.delete_from_timeseries(turb_data, "Data/Misc/delete_date_ranges.txt")

fDOM_raw_data = dp.interpolate_missing_intervals(fDOM_raw_data)
stage_data = dp.interpolate_missing_intervals(stage_data)
turb_data = dp.interpolate_missing_intervals(turb_data)


def compare_edge(edge_1: np.ndarray, edge_2: np.ndarray) -> int:
    """
    Compare edge_1 to edge_2 to see if edge_1 intersects
    with edge_2, appears before edge_2, or appears after edge_2

    edge_1 : edge in timeseries represented by start and end timestamp (julian time)
    edge_2 : edge in timeseries represented by start and end timestamp (julian time)
    return : -1 == edge_1 comes before edge_2
              0 == edge_1 intersects with edge_2
              1 == edge_1 comes after edge_2
    """
    # If start or end of edge_1 is within/equal start and end of edge_2, then match
    # If start e1 before start e2 and end e1 after end e2, then match
    if (
        (edge_1[0] >= edge_2[0] and edge_1[0] <= edge_2[1])
        or (edge_1[1] >= edge_2[0] and edge_1[1] <= edge_2[1])
        or edge_1[0] < edge_2[0]
        and edge_1[1] > edge_2[1]
    ):
        return 0
    # If end of edge_1 is before or equal to start of edge_2, then before
    elif edge_1[1] <= edge_2[0]:
        return -1
    elif edge_1[0] >= edge_2[1]:
        return 1
    else:
        print(dp.julian_to_datetime(edge_1[0]), " ", dp.julian_to_datetime(edge_1[1]))
        print(dp.julian_to_datetime(edge_2[0]), " ", dp.julian_to_datetime(edge_2[1]))
        raise IndexError("No determination made on edge")


# 2012-05-16 16:30:00   2012-05-16 17:30:00
# 2012-05-16 16:45:00   2012-05-16 17:15:00
def event_binary_search(single_edge, list_edges):
    """
    Given a single edge and a sorted list of edges,
    use binary search like algorithm to find a possible
    match if it exists
    """
    if list_edges.shape[0] == 0:
        return False
    mid = math.ceil(len(list_edges) / 2) - 1
    result = compare_edge(single_edge, list_edges[mid])
    if result == 0:
        return True
    elif result == 1:
        return event_binary_search(single_edge, list_edges[mid + 1 : len(list_edges)])
    else:
        return event_binary_search(single_edge, list_edges[0:mid])


def determine_stage_rise_metric(list_de, list_gt):
    """
    Given a list of detected edges(de) and a list of ground truth edges(gt)
    determine the accuracy of the detected edges... Not sure by what metric yet
    A de can either be a TP or FP. If a gt edge has no corresponding de edge, then
    the gt edge becomes a FN in the de list. NLog(N) time complexity

    list_de: list of detected edges to compare to ground truth
    list_gt: list of ground truth edges by which to judge the detected edges

    return: dict containing: overall accuracy of detected edges
                             list_de with determinations attached to each edge (TP,FP)
                             list_gt with determination attached to each edge (FN, NFN(Not False Negative))
    """

    # Determine FP,TP in detected edges
    de_determinations = []
    for i, edge in enumerate(list_de):
        result = event_binary_search(edge, list_gt)
        if result:
            de_determinations.append("TP")
        else:
            de_determinations.append("FP")

    # Determine FN in detected edges
    gt_determinations = []
    for i, edge in enumerate(list_gt):
        result = event_binary_search(edge, list_de)
        if result:
            gt_determinations.append("NFN")
        else:
            gt_determinations.append("FN")

    # Use determinations to calculate accuracy metric of some sort
    FP = de_determinations.count("FP")
    TP = de_determinations.count("TP")
    FN = gt_determinations.count("FN")

    FN_weight = 1.0
    FP_weight = 1.0

    metric = TP / (TP + (FN * FN_weight) + (FP * FP_weight))

    return {
        "metric": metric,
        "de_determ": de_determinations,
        "gt_determ": gt_determinations,
        "FP": FP,
        "TP": TP,
        "FN": FN,
    }


def main():
    # Read in labeled data
    gt_fname = "Data/manual_annotating_data/stage_rise/stage_rises_0k-300k.csv"

    gt_data = []
    with open(gt_fname, "r", newline="") as gt_file:
        reader = csv.reader(gt_file, delimiter=",")
        for row in reader:
            gt_data.append(
                (
                    dp.datetime_to_julian(
                        datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    ),
                    dp.datetime_to_julian(
                        datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                    ),
                )
            )

    gt_data = np.array(gt_data)

    iterations = 2  # based on the confidence of parameters optimality desired

    # Hyper params based on reasonable estimations
    lag_bounds = (1, 20)
    threshold_bounds = (0, 20)
    influence_bounds = (0, 1)

    best_params = {"lag": 0, "threshold": 0, "influence": 0}

    max_acc = 0
    max_result = None
    for i in range(iterations):

        # Randomize hyper params within set bounds
        lag = np.random.randint(lag_bounds[0], lag_bounds[1] + 1)
        threshold = np.random.uniform(threshold_bounds[0], threshold_bounds[1])
        influence = np.random.uniform(influence_bounds[0], influence_bounds[1])

        # Generate stage rises
        stage_edges = detect_edges(stage_data[:, 1], lag, threshold, influence)[
            "signals"
        ]
        # Process into start and end timestamps for each edge
        rise_indices = []
        for j in range(len(stage_edges)):
            if stage_edges[j] == 1:
                rise_indices.append(j)
        starts = [stage_data[rise_indices[0], 0]]
        ends = []
        for j, indx in enumerate(rise_indices):
            # If this condition is true, then this point is the end of an edge(it can also be the start in case of 1 data point edge)
            if (j < len(rise_indices) - 1) and rise_indices[j + 1] - 1 != indx:
                # Designate this point as an end and next point at a start, but also evaluate next point as it could be both a start and an end
                ends.append(stage_data[indx, 0])
                starts.append(stage_data[rise_indices[j + 1], 0])
        ends.append(stage_data[rise_indices[-1], 0])

        de_data = np.transpose(np.array((starts, ends)))

        # Algo to compare generated to ground truth
        result = determine_stage_rise_metric(de_data, gt_data)
        metric = result["metric"]

        if i % 10 == 0:
            print(
                "{} Metric: {:.4f}   Lag: {}   Influence: {:.4f}   Threshold: {:.4f}   Max Acc: {:.4f}".format(
                    i, metric, lag, influence, threshold, max_acc
                )
            )
            print(
                "TP Rate: {}/{}   FP Rate: {}/{}   FN Rate: {}/{}".format(
                    result["TP"],
                    len(starts),
                    result["FP"],
                    len(starts),
                    result["FN"],
                    gt_data.shape[0],
                )
            )

        if metric > max_acc:
            max_acc = metric
            best_params["lag"] = lag
            best_params["threshold"] = threshold
            best_params["influence"] = influence
            max_result = result

            print("New Max:")
            print(
                "{} Metric: {:.4f}   Lag: {}   Influence: {:.4f}   Threshold: {:.4f}   Max Acc: {:.4f}".format(
                    i, metric, lag, influence, threshold, max_acc
                )
            )
            print(
                "TP Rate: {}/{}   FP Rate: {}/{}   FN Rate: {}/{}".format(
                    result["TP"],
                    len(starts),
                    result["FP"],
                    len(starts),
                    result["FN"],
                    gt_data.shape[0],
                )
            )
            print("\n")
    # Write out data to disk
    out_fname = "./Stage_Rise_Hyperparam_Opt.txt"

    # Store non list/array data in txt
    with open(out_fname, "w") as out_file:
        out_file.write("Max Metric Achieved: {:.3f}\n".format(max_acc))
        out_file.write(
            "Lag: {} , Influence: {:.6f} , Theshold: {:.6f}\n".format(
                best_params["lag"], best_params["influence"], best_params["threshold"]
            )
        )
        out_file.write(
            "TP Rate: {}/{}   FP Rate: {}/{}   FN Rate: {}/{}\n".format(
                max_result["TP"],
                max_result["TP"] + max_result["FP"],
                max_result["FP"],
                max_result["TP"] + max_result["FP"],
                max_result["FN"],
                gt_data.shape[0],
            )
        )
    # Pickle list/array data
    results_dict = {
        "de_determ": max_result["de_determ"],
        "gt_determ": max_result["gt_determ"],
        "de_data": de_data,
        "gt_data": gt_data,
    }
    with open("./Stage_Rise_Hyperparam_Pickle.pkl", "wb") as pck_file:
        pickle.dump(results_dict, pck_file)


if __name__ == "__main__":
    main()
