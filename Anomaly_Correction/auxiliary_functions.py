"""
This module contains the following functions: 

- extract_runoff
- get_stage_events 
- detect_edges

"""

import numpy as np
import copy
import sys
sys.path.insert(1,'../')
from Tools.data_processing import smooth_data, find_turning_points
from scipy.signal import find_peaks

def extract_runoff(data_in : np.ndarray,
                   min_diff : float,
                   return_ratio : float,
                   start_slope : float,
                   end_slope : float,
                   smooth_constant : int,
                   min_dur : int = 0,
                   dyslp : float =.001) -> list[np.ndarray]:
    """
    Function extracts runoff events from data given hyper parameters
    Delineates what we refer to as "normal peaks"

    data_in:         timeseries containing runoff events
    min_diff:        minimum threshold difference between start/end value and peak value of an event
    return_ratio:    ratio of baseline that even must return to be considered over; determines where an event terminates 
    start_slope:     threshold to cut off flat starts
    end_slope:       threshold to cut off flat ende
    smooth_constant: how many times we apply a smoothing filter to the data before detecting events
    min_dur:         minimum duration that an event has to exceed
    dyslp:           a dynamic slope threshold used to cut off flat starts and ends

    return:          list of events detected 
    """
    RETURNCONSTANT = min_diff / 3

    # Smooth Data
    data = copy.deepcopy(data_in)
    data[:, 1] = smooth_data(data[:, 1], smooth_constant)

    # Identify and extract Turning Points (peaks and valleys)

    turning_points = find_turning_points(data[:, 1])
    indices = turning_points[:, 0].astype(int)
    stack_data = data[indices, 1].reshape(len(indices),
                                          1)  # extract data to stack
    turning_points = np.hstack(
        (turning_points,
         stack_data))  # stack 3rd column that is value at point
    # Stack entry for first element in the data if it is 'very low', making it a valley
    if turning_points[0, 1] == 1 and data[0, 1] < np.mean(data[:, 1]) / 10:
        turning_points = np.vstack((np.array([0, 0, data[0,
                                                         1]]), turning_points))
    # Stack entry for last element if it is 'very low' making it a valley
    if turning_points[-1, 1] == 1 and data[-1, 1] < np.mean(data[:, 1]) / 10:
        turning_points = np.vstack(
            (turning_points, np.array([data.shape[0] - 1, 0, data[-1, 1]])))

    # Remove incomplete events
    while turning_points[0, 1] == 1:
        turning_points = np.delete(turning_points, 0, 0)
    while turning_points[-1, 1] == 1:
        turning_points = np.delete(turning_points, -1, 0)

    # Identify start and end points of each event
    peak_valley_diffs = np.diff(turning_points[:, 2])
    peak_valley_diffs = np.append(peak_valley_diffs,
                                  0).reshape(len(peak_valley_diffs) + 1, 1)
    turning_points = np.hstack((turning_points, peak_valley_diffs))
    np.set_printoptions(suppress=True)

    i = 0  # step through data
    count = 0
    is_complete = True
    num_inflect = turning_points.shape[0]
    starts = []
    ends = []
    while i < num_inflect-1:
        j = 1
        diff = turning_points[i, 3] + turning_points[i + j, 3]
        # Until data has returned
        while diff > max(return_ratio * max(abs(turning_points[i:i + j+1, 3])),RETURNCONSTANT):
            if i + j < num_inflect - 1:
                j +=1
                diff = diff + turning_points[i + j, 3]
            else:
                is_complete = False
                break
        starts.append(i)
        ends.append(i + j)
        i = i + j + 1
        count += 1

    if is_complete == 0:
        starts = starts[0:-1]
        ends = ends[0:-1]

    # Extract events and put them into an individual entry
    num_event = 0
    events = []
    dyslp_base = dyslp; 
    for i in range(len(starts)):
        dyslp = dyslp_base
        date = data_in[int(turning_points[starts[i],0]):int(turning_points[ends[i]+1, 0])+1,0]
        event_smooth = data[int(turning_points[starts[i],0]):int(turning_points[ends[i]+1,0])+1, 1]
        event = data_in[int(turning_points[starts[i],0]):int(turning_points[ends[i]+1, 0])+1,1]

        event_flow = np.hstack((date.reshape(len(date),1), event.reshape(len(event), 1),event_smooth.reshape(len(event_smooth), 1)))

        temp1 = np.diff(event_flow[:, 2])
        # Select events whose peaks exceed min_diff
        if max(event_flow[:, 1]) - event_flow[0, 1] > min_diff and max(event_flow[:, 1]) - event_flow[-1, 1] > min_diff:
            dyslp = dyslp * np.ptp(event_flow[:, 1])  # dynamic slope threshold
            # Shorten events by removing flat starts and end
            while temp1.size > 0 and temp1[0] < min(start_slope, dyslp):
                event_flow = event_flow[1:,:]
                temp1 = temp1[1:]
            while temp1.size > 0 and temp1[-1] > -min(end_slope, dyslp):
                event_flow = event_flow[0:-1, :]
                temp1 = temp1[0:-1]

            # Check slope of original flow data
            if temp1.size > 0:
                temp2 = np.diff(event_flow[:, 1])
                while temp2.size > 0 and temp2[0] <= min(start_slope, dyslp):
                    event_flow = event_flow[1:, :]
                    temp2 = temp2[1:]
                while temp2.size > 0 and temp2[-1] >= -min(end_slope, dyslp):
                    event_flow = event_flow[0:-1, :]
                    temp2 = temp2[0:-1]

                # Select events whose duration exceeds min_dur
                if temp2.size > min_dur:
                    num_event += 1
                    events.append(event_flow[:, 0:2])
    return events


def get_stage_events(stage_data):
    prominence_range = [.01,None]
    width_range = [None,None]
    wlen = 500
    distance = 1
    rel_height =1.0
    stage_peaks, props = find_peaks(stage_data[:,1],
                            height = (None, None),
                            threshold = (None,None),
                            distance = distance,
                            prominence = prominence_range,
                            width = width_range,
                            wlen = wlen,
                            rel_height = rel_height)
    stage_events = []

    for peak in stage_peaks:
        stage_events.append(np.array((stage_data[peak-1], stage_data[peak], stage_data[peak+1])))
    return stage_events

def detect_edges(y : np.ndarray, lag : int, threshold : float, influence : float) -> dict:
    """
    Given a timeseries, detect all rising and falling edges using given hyperparameters 

    y:         timeseries to detect edges in 
    lag:       window size of previous data points to calculate mean and std from
    threshold: number of standard deviations away from mean at which a datapoint is a rising or falling edge
    influence: influence of new data points on mean and std 
    
    return:    a dictionary containing: the signal (-1 for falling edges, 0, 1 for rising edge) for each data point 
                                        mean of moving window at each data point 
                                        std of moving window at each data point 
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))