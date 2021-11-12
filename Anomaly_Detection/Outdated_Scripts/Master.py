"""
HYPERPARAMETERS - changing these will change how events are detected
"""
slope_ratio = 1  # used in 'flag_runoffs'

# fDOM hyperparams 

fDOM_peak_threshold = 3 # the threshold above 'baseline' at which the peak is considered to be an event
fDOM_return_ratio = .90 # event must return to specified percent of basleine to be considered 'over'
fDOM_start_slope = .001 # this threshold cuts off the flat starts of events 
fDOM_end_slope = .0001 # this theshold cuts off the flat ends of events 
fDOM_smooth_passes = 4 # the number of smoothing passes applied to data before event detection 
fDOM_min_dur = 5 # minimum number of samples a normal peak must span
# stage hyperparams 

stage_peak_threshold = .02
stage_return_ratio = .6
stage_start_slope = .001 
stage_end_slope = .0001 
stage_smooth_passes = 4 
stage_min_dur = 5 # minimum number of samples a normal peak must span

# turbidity hyperparams 

turb_peak_threshold = 15 
turb_return_ratio = .6 
turb_start_slope = .001 
turb_end_slope = .0001 
turb_smooth_passes = 4
"""
START LOADING DATA - Load converted data (no events, just a time and vale)
"""

# Imports and Helper functions
import scipy.io as sio
from scipy.signal import find_peaks
from os.path import dirname, join as pjoin
import numpy as np
import sys
import copy

from auxiliary_functions import extract_runoff, flag_runoffs

import numpy as np
import csv
import datetime
import sys
import math

def datetime_to_julian(date : datetime.datetime) -> float:
    """
    Convert datetime object to julian time using algorithm outlined by Fliegel and van Flandern (1968):
    
    date:   date to convert

    return: julian time equivalent
    """
    interm1 = math.floor((14-date.month)/12)
    interm2 = date.year + 4800 - interm1
    interm3 = date.month + 12*interm1 - 3

    jdn = (date.day + math.floor((153*interm3 + 2)/5) + 365*interm2 + 
           math.floor(interm2/4) - math.floor(interm2/100) + math.floor(interm2/400) - 32045)

    jd = jdn + (date.hour - 12) / 24 + date.minute / 1440 + date.second / 86400 + date.microsecond / 86400000000
    return jd


def julian_to_datetime(date : float) -> datetime.datetime:
    """
    Convert julain to datetime object using algorithm outlined by Fliegel and van Flandern (1968)
    
    date:   julian time

    return: equivalent date as python datetime object
    """
    date, fraction = math.floor(date + 0.5), date + 0.5 - math.floor(date + 0.5)

    interm_val1 = date+68569
    interm_val2 = 4*interm_val1//146097
    interm_val1 = interm_val1-(146097*interm_val2+3)//4
    year = 4000*(interm_val1+1)//1461001
    interm_val1 = interm_val1-1461*year//4+31
    month = 80*interm_val1//2447
    day = interm_val1-2447*month//80
    interm_val1 = month//11
    month = month+2-12*interm_val1
    year = 100*(interm_val2-49)+year+interm_val1

    year = int(year)
    month = int(month)
    day = int(day)

    fractional_component = int(fraction * (1e6*24*3600))

    hours = int(fractional_component // (1e6*3600))
    fractional_component -= hours * 1e6*3600

    minutes = int(fractional_component // (1e6*60))
    fractional_component -= minutes * 1e6*60

    seconds = int(fractional_component // 1e6)
    fractional_component -= seconds*1e6

    fractional_component = int(fractional_component)

    date = datetime.datetime(year=year, month=month, day=day, hour=hours, minute=minutes, second=seconds)

    if date.second == 59: 
        date += datetime.timedelta(seconds = 1)
    return date

def trim_timeseries(data : np.ndarray, start_time : float, end_time : float) -> np.ndarray:
    """
    Trim given timeseries to within give julian timestamps 

    data:       timeseries
    start_time: earliest allowed timestamp; in julian time
    end_time:   last allowed timestamp; in julian time

    return:     numpy array
    """
    trimmed_data = np.zeros((len(data),2))
    index = 0
    for i in range(len(data)):
        if data[i,0] <= end_time and data[i,0] >= start_time:
            trimmed_data[index,0] = data[i,0]
            trimmed_data[index,1] = data[i,1]
            index +=1
    trimmed_data = trimmed_data[0:index,:]
    return trimmed_data

def delete_from_timeseries(data : np.ndarray, dates_file : str) -> np.ndarray:
    """
    Uses provided csv file of start and end time pairs and deletes
    all data from timeseries within those ranges

    data:       timeseries 
    dates_file: file path 

    return:     filtered timeseries
    """
    return_data = np.zeros((len(data),2))
    index = 0

    with open(dates_file,newline='') as file:
        reader = csv.reader(file, delimiter = ',')
        time_list = []
        for row in reader:
            time_list.append([datetime_to_julian(datetime.datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S')),
                              datetime_to_julian(datetime.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S'))])

        for i in range(data.shape[0]):
            flag = True 
            time = data[i,0]
            for row in time_list:
                if time >= row[0] and time <= row[1]:
                    flag = False 
            if flag:
                return_data[index,0] = data[i,0]
                return_data[index,1] = data[i,1]
                index +=1

    # Remove excess data 
    return return_data[~np.all(data == 0, axis=1)]


def remove_5_min_intervals(data: np.ndarray, is_julian: bool = False , convert_to_array: bool = False):
    """
    Remove all samples that are not in 15 minute intervals and correct 'off by one second' errors 

    data:             timeseries
    is_julian:        are the timestamps in julian time? 
    convert_to_array: should the return type be a list of numpy arrays or a 2D numpy array? 

    return:           cleansed timeseries -> ist of numpy arrays or a 2D numpy array depending
                      on input param
    """
    # Convert to datetime 
    if is_julian:
        data = [[julian_to_datetime(entry[0]),entry[1]] for entry in data]
    else: 
        data = [[datetime.datetime.strp(entry[0],'%Y-%m-%d %H:%M:%S'),entry[1]] for entry in data]
        
    # Remove violating entries 
    valid_min = [0,15,30,45]
    time_delta = datetime.timedelta(seconds = 1)
    return_data = []
    
    for entry in data:
        if entry[0].second != 0:
            entry[0]+=time_delta
        if entry[0].minute in valid_min:
            if is_julian:
                entry[0] = datetime_to_julian(entry[0])
            return_data.append(entry)
        else:
            print(entry)
    
    if convert_to_array:
        return_data = np.array(return_data)
        
    return return_data


def flip_timeseries(data : np.ndarray) -> np.ndarray:
    """
    Flip the timeseries given; done in place
    
    data:   timeseries 

    return: flipped timeseries
    """
    for i in range(data.shape[1]):
        if data[i, 1] < 0:
            data[i, 1] = 0
    max_raw = max(data[:, 1])

    for i in range(data.shape[1]):
        data[i, 1] = max_raw - data[i, 1]
    return data

def interpolate_timeseries(data : np.ndarray,  pks : np.ndarray, locs : np.ndarray, w : np.ndarray, p : np.ndarray) -> np.ndarray:
    """
    Interpolate the timeseries given peaks, locations of peaks, width, and prominence

    data:   timeseries   
    pks:    peak heights 
    locs:   peak locations as indexes within data
    w:      widths or peaks
    p:      prominences of peaks

    return: timeseries interpolated with given peak information
    """
    for i in range(locs.shape[1]):
        try:
            half = np.floor(w[i] / 2)
            # half = 20
            start_index = locs[i] - half
            end_index = locs[i] + half
            start_val = data[start_index, 1]
            end_val = data[end_index, 1]
            len = data[start_index:end_index + 1, 1].shape[1]
            data[start_index:end_index + 1,
                   1] = np.linspace(start_val, end_val, len)
        except:
            pass

    return data

def add_flags(data : np.ndarray , flag : str= '') -> list[list]:
    """
    Add given flags to data

    data:   data to add flags to 
    flag:   flag to add 

    return: data with flags added
    """
    return [[row[0],row[1],flag] for row in data]

def correct_one_second_error(data):
    """
    Round all timestamps where seconds == 59 
    to seconds == 0, causing the minute to +=1

    data:   timeseries to correct -> list or ndarray depending 

    return: corrected timeseries  -> list of ndarray depedning on input 
    """
    time_delta = datetime.timedelta(seconds=1)
    for row in data: 
        if row[0].second !=0:
            row[0] = row[0] + time_delta
    return data

def smooth_data(data, passes):
    """
    Function smooths data using a moving average filter window of size 3 
    data = the data to be smoothed (a column of data)
    passes = how many smoothing passes to apply to the data
    return - smoothed data column of same length
    """
    for n in range(passes):  # if passes == 0, then no smoothing will be applied
        for i in range(1, len(data) - 1):
            data[i] = (data[i - 1] + 2.0 * data[i] + data[i + 1]) / 4.0  # weight unsmoothed value 2x
    return data

def find_turning_points(data):
    """
    Find 'turning points' in the data 
    turning points are peaks and valleys in the data 
    data = column of data to be analyzed 
    return - two col array of turning points where:
        column 1 : the index of turning point in the data
        column 2 : a binary label of peak = 1 or valley = 0
    """
    data_diffs = np.diff(data)  # difference between each consecutive pair of values
    sign_diffs = np.diff(np.sign(data_diffs))  # difference between signs of data_diffs
    idx = np.argwhere(sign_diffs != 0)  # find indexs of turning points

    turning_points = np.hstack((idx + 1, sign_diffs[idx]))  # stack index of turning points with sign differences
    for i in range(len(turning_points)):
        # Mark as a valley
        if turning_points[i, 1] > 0:
            turning_points[i, 1] = 0
        # Mark as a peak
        elif turning_points[i, 1] < 0:
            turning_points[i, 1] = 1
    return turning_points

def compute_event_info(event : np.ndarray) -> dict:
    """
    Return a structure with start,end,peak,centroid of an event

    event:  single event or anomaly
    
    return: information of event 
    """
    _, iPeak = max(event[:, 1])

    struct = {}
    struct['start'] = event[0, 0]
    struct['end'] = event[-1, 0]
    struct['peak'] = event[iPeak, 0]
    struct['centroid'] = sum(np.multiply(event[:, 0], event[:, 1])) / sum(
        event[:1])
    return struct

def julian_slope_ratio(event : np.ndarray) -> float:
    """
    Compute return ratio of left side of event over right side of event

    event:  single evnet of anonaly

    return: slope ratio of event
    """
    metadata = compute_event_info(event)
    peak_val = max(event[:, 1])
    start_val = event[0, 1]
    end_val = event[-1, 1]

    rising_slope = (peak_val - start_val) / (metadata['peak'] -
                                             metadata['start'])
    falling_slope = (peak_val - end_val) / (metadata['end'] - metadata['peak'])

    return rising_slope / falling_slope

def interpolate_flagged_events(events : np.ndarray, data : np.ndarray) -> np.ndarray:
    """
    Interpolate flagged events

    events: array of detected events  
    data:   timeseries 

    return: interpolated data 
    """
    for i in range(events.shape[0]):
        event = events[i][0]
        flag = events[i][1]

        if flag == 'invalid':
            start = event[0, 0]
            end = event[-1, 0]
            _, peak = max(event[:, 1])
            idx = data[:, 0] >= start and data[:, 0] <= end
            start_val = event[0, 1]
            end_val = event[-1, 1]

            event[:, 1] = np.linspace(start_val, end_val, event.shape[1])
            data[idx, :] = event
        events[i] = event
    return data

def read_in_timeseries(in_file : str, is_julian : bool = False) -> list[list]:
    """
    Read timeseries in from csv to list 

    in_file:   file path
    is_julian: are the timestamps in julian time? 

    return:    timeseries
    """
    data = []
    with open(in_file, 'r', newline ='') as file:
        reader = csv.reader(file, delimiter= ',')
        for row in reader:
            if is_julian:
                data.append([float(row[0]),float(row[1])])
            else: 
                data.append([row[0],float(row[1])])
        file.close()
    return data


def write_data_to_trainset(fDOM_data : np.ndarray, 
                           stage_data : np.ndarray,
                           turb_data : np.ndarray, 
                           out_file : str, 
                           has_flags : bool = False, 
                           is_julian_time : bool = True, 
                           data_start : int = 0, 
                           data_end : int =  sys.maxsize) -> None:
    """ 
    Function takes in 3 timeseries, adjusts and aligns timeseires,
    combines data, formats to Trainset specification, and writes 
    out combined dataset to given out directory location in csv format
    
    *_data:         timeseries
    out_file:       file path name to write data out to
    has_flags:      does the data have flags or not?
    is_julian_time: is data in julian time ?, else datetime
    data_start:     first index in data to start writing from
    data_end:       last index in data to write up to 
    """
    # Add empty flags if applicable 
    if not has_flags:
        fDOM_data = add_flags(fDOM_data)
        stage_data = add_flags(stage_data)
        turb_data = add_flags(turb_data)
    
    # Convert from julian time to datetime objects
    if is_julian_time:
        fDOM_data = [[julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in fDOM_data]
        stage_data = [[julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in stage_data]
        turb_data = [[julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in turb_data]
        
    # Adjust data that is off by one second
    fDOM_data = correct_one_second_error(fDOM_data)
    stage_data = correct_one_second_error(stage_data)
    turb_data = correct_one_second_error(turb_data)
    
    # Align data to fDOM
    stage_filtered = []
    turb_filtered = []
    
    for i in range(len(fDOM_data)):
        fDOM_time = fDOM_data[i][0]
        
        j = i
        while fDOM_time != stage_data[j][0]:
            j+=1
        stage_filtered.append(stage_data[j])
        
        j = i
        while fDOM_time != turb_data[j][0]:
            j+=1
        turb_filtered.append(turb_data[j])
        
        # Raise error if data still not in align
        if not (fDOM_time == stage_filtered[i][0] == turb_filtered[i][0]):
            raise ValueError("Data align failed at idx: {} ,with fDOM: {}, stage: {}, turb: {}".format(i,fDOM_time, stage_filtered[i][0],turb_filtered[i][0]))
            
    stage_data = stage_filtered
    turb_data = turb_filtered
    
    # Write out data to csv 
    with open(out_file, 'w',newline='') as outfile:
        writer = csv.writer(outfile,delimiter=',')
        writer.writerow(['series', 'timestamp', 'value', 'label'])
        
        for i in range(len(fDOM_data)):
            # Skip entries not in given range
            if i >= data_start and i < data_end:
                # Convert to ISO8061 wformat 
                fDOM_time = fDOM_data[i][0].isoformat() + '.000Z'
                stage_time = stage_data[i][0].isoformat() + '.000Z'
                turb_time = turb_data[i][0].isoformat() + '.000Z'
                
                writer.writerow(['fDOM', fDOM_time, fDOM_data[i][1], fDOM_data[i][2]])
                writer.writerow(['Stage', stage_time, stage_data[i][1], stage_data[i][2]])
                writer.writerow(['Turbidity',turb_time, turb_data[i][1], turb_data[i][2]])
    outfile.close()

def collapse_events(data : np.ndarray) -> list[np.ndarray]:
    """
    Collapse events to one continuous list of data points
    Adjsut events that share the same start and end

    data:   events to collapse, julian time

    return: collapsed events
    """
    return_list = []
    time_delta = datetime.timedelta(minutes=15)
    for i,event in enumerate(data):
        # Events should not share starting and end points that are the same of adjacent
        indx=0
        if i != 0:
            end_prev = julian_to_datetime(return_list[-1][0])
            start_curr = julian_to_datetime(event[0][0])
            if end_prev == start_curr:
                indx=2
            elif (end_prev + time_delta) == start_curr:
                indx=1
        for j in range(indx,event.shape[0]):
            return_list.append(event[j])
    return return_list

def merge_data(data : np.ndarray, events : list[np.ndarray], flag : str, default_flag : str) -> list[list]:
    """
    Merge given events into data such that event flags preceed default flags

    data:         timeseries with or without flags       
    events:       collapsed event list where each entry 
                  in the list is a nump array of 
                  [timestamp, value]                      
    flag:         value of flag to be applied to entries
                  that are a part of an event             
    default_flag: value of flag to be applies to entries
                  not part of an event                    
    
    return:       timeseries with addition of 3rd column
                  of flags for each data point
    """
    merged = []
    index = 0

    for i in range(len(data)):
        
        if index < len(events) and data[i,0] == events[index][0]:
            merged.append([events[index][0], events[index][1], flag])
            index+=1
        else: 
            merged.append([data[i,0],data[i,1], default_flag])
        
    return merged

def write_detected_to_trainset(fDOM_events : list[np.ndarray], 
                               stage_events: list[np.ndarray],  
                               turb_events: list[np.ndarray],  
                               fDOM_data : np.ndarray, 
                               stage_data : np.ndarray, 
                               turb_data : np.ndarray, 
                               flags : dict, 
                               out_file : str,
                               data_start : int = 0,
                               data_end : int = sys.maxsize) -> None:
    """
    Function merges events/anomalies into entire timeseres: Entry has special flag if part of an event, else has default flag.
    Merged data is then written out to csv file using function write_data_to_trainset. 
    
    *_events:   events for each timeseries, data can be empty. 
    *_data:     timeseries data 
    flags:      dictionary of 6 flags to apply to data
    out_file:   file path name to write data out to
    data_start: first index in data to start writing from
    data_end:   last index in data to write up to 
    """

    # Collapse event lists down to one list of every point that is a part of a peak 
    fDOM_events = collapse_events(fDOM_events)
    stage_events = collapse_events(stage_events)
    turb_events = collapse_events(turb_events)
    
    # with open('./test5.csv', 'w', newline = '') as outfile:
    #     writer = csv.writer(outfile, delimiter = ',')
    #     for entry in fDOM_events:
    #         writer.writerow([str(julian_to_datetime(entry[0])), entry[1]])
    #     outfile.close()
    
    # max = 0
    # for i in range(1,len(fDOM_events)):
    #     if not fDOM_events[i][0] > fDOM_events[i-1][0]:
    #         print("Error at",i)
    #         print(fDOM_data[i],fDOM_data[i-1])

    # Merge each timeseries with events
    fDOM_merged = merge_data(fDOM_data, fDOM_events, flags['fDOM_flag'], flags['default_fDOM_flag'])
    stage_merged = merge_data(stage_data, stage_events, flags['stage_flag'], flags['default_stage_flag'])
    turb_merged = merge_data(turb_data, turb_events, flags['turb_flag'], flags['default_turb_flag'])

    # with open('./test2.csv', 'w', newline = '') as outfile:
    #     writer = csv.writer(outfile, delimiter = ',')
    #     for entry in fDOM_merged:
    #         writer.writerow(entry)
    #     out_file.close()


    write_data_to_trainset(fDOM_merged, stage_merged, turb_merged, out_file, True, True, data_start, data_end)




fDOM_raw_data = np.array(read_in_timeseries('/Users/zachfogg/Desktop/DB-SRRW/Data/converted_data/julian_format/fDOM_raw_10.1.2012-9.4.2020.csv',True))
stage_data = np.array(read_in_timeseries('/Users/zachfogg/Desktop/DB-SRRW/Data/converted_data/julian_format/stage_10.1.11-1.1.19.csv',True))
turb_data = np.array(read_in_timeseries('/Users/zachfogg/Desktop/DB-SRRW/Data/converted_data/julian_format/turbidity_raw_10.1.2011_9.4.2020.csv',True))

# Jonah chose to scale the values down to 22.4% or original; this shifts forward 15 minutes
# fDOM_data[:, 0] = fDOM_data[:, 0] + (.224 * 1)

# Trim timeseries to all span the same length of time
start_time = fDOM_raw_data[0,0]
end_time = stage_data[-1,0]
turb_data = trim_timeseries(turb_data,start_time, end_time)
fDOM_raw_data = trim_timeseries(fDOM_raw_data, start_time, end_time)
stage_data = trim_timeseries(stage_data,start_time, end_time)

"""
START PREPROCESSING / DESPIKING
- Here we make use of helper functions (including hydrun) to extract events and timestamps
"""

# Extract events for stage
stage_events, num_stage_events = extract_runoff(stage_data, stage_peak_threshold,
                                                stage_return_ratio, stage_start_slope,
                                                stage_end_slope,
                                                stage_smooth_passes,
                                                min_dur = stage_min_dur)

# Extract event for turbidity
turb_events, num_turb_events = extract_runoff(turb_data, turb_peak_threshold,
                                              turb_return_ratio, turb_start_slope,
                                              turb_end_slope,
                                              turb_smooth_passes)

# Extract events for fDOM 
fDOM_events, num_fDOM_events = extract_runoff(fDOM_raw_data, fDOM_peak_threshold,
                                              fDOM_return_ratio, fDOM_start_slope,
                                              fDOM_end_slope,
                                              fDOM_smooth_passes,
                                              min_dur = fDOM_min_dur)
# These determine what labels the events will be assigned
flags = {
    "fDOM_flag" : "normal_peak",
    "default_fDOM_flag" : "",
    "stage_flag" : "normal_peak",
    "default_stage_flag" : "",
    "turb_flag" : "normal_peak",
    "default_turb_flag" : ""
}

"""
This module contains functions that were created for tasks related
to data processing. This module contains the following functions: 

- trim_timeseries
- delete_from_timeseries
- remove_5_min_intervals 
- merge_data
- flip_timeseries
- interpolate_timeseries
- julian_to_datetime 
- datetime_to_julian
- collapse_events
- add_flags
- correct_one_second_errors
- smooth_data
- find_turning_points
- compute_event_info
- julian_slope_ratio
- interpolate_flagged_events
"""

write_detected_to_trainset(fDOM_events, 
                        stage_events, 
                        turb_events, 
                        fDOM_raw_data,
                        stage_data, 
                        turb_data,
                        flags,
                        '/Users/zachfogg/Desktop/DB-SRRW/Data/anomaly_data/Trainset_Compliant/normal_peaks100k.csv',
                        0,10000)

# # Extract runoff events from fDOM
# fDOM_events, num_fDOM_events = flag_runoffs(fDOM_raw_data, fDOM_PKThreshold,
#                                             fDOM_ReRa, slope_ratio,
#                                             stage_events, num_stage_events,
#                                             turb_events, num_turb_events,
#                                             fDOM_start_slope, fDOM_end_slope,
#                                             fDOM_smooth_coeff)

# fDOM_despiked = dp.interpolate_flagged_events(fDOM_events, num_fDOM_events,
#                                            fDOM_raw_data)

# # Flip fDOM data upside down to get rid of negative spikes
# flipped_fDOM = dp.flip_timeseries(fDOM_despiked)
# # peaks: peak values, locations: indices of peaks
# locations, props = find_peaks(flipped_fDOM[:, 1],
#                               height=min_flipped_height,
#                               prominence=0,
#                               width=0)
# peaks = props['peak_heights']
# widths = props['widths']
# prominences = props['prominences']
# fDOM_despiked = dp.interpolate_timeseries(fDOM_despiked, peaks, locations, widths,
#                                        prominences)
# """
# START PLOTTING
# """

# First subplot is raw vs despiked fDOM, with some types of spikes flagged
