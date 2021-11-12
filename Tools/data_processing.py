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
- interpolate_missing_intervals
- delete_missing_data_events
- align_stage_to_fDOM
- merge_additional_data
- stage_rises_to_data
- delete_missing_data_peaks

"""

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
    start_index = -1
    end_index = -1
    for i in range(len(data)):
        if data[i,0] == start_time:
            start_index = i
            break
    for i in range(start_index, len(data)):
        if data[i,0] == end_time: 
            end_index = i
            break
    if start_index != -1 and end_index != -1:
        return data[start_index : end_index+1,:]

    else: 
        raise IndexError("Start time or end time not in data")

def interpolate_missing_intervals(data: np.ndarray) -> np.ndarray:
    """
    Fill missing data gaps that are of 5 or less data points by interpolating accross 

    data:   timeseries to check for missing data gaps
    return: timeseries with qualifying gaps interpolated accross
    """
    delta15 = datetime.timedelta(minutes=15)
    insert_vals = []
    insert_indxs = []
    for i in range(data.shape[0]-1):
        # Establish that there is at least 1 missing value
        if julian_to_datetime(data[i,0]) + delta15 != julian_to_datetime(data[i+1,0]): 
            # Establish length of gap and only interpolate accross gaps of less than 5. 
            diff = int((((julian_to_datetime(data[i+1,0]) - julian_to_datetime(data[i,0])).total_seconds())/900)-1)
            if diff < 5:
                start = data[i,1]
                end = data[i+1,1]
                increment = (end - start) / (diff + 1) 
                # Insert values for each missing value
                k = 1
                for j in range(i,i+diff):
                    increment_timedelta = datetime.timedelta(minutes = (15 * k))
                    insert_vals.append([datetime_to_julian(julian_to_datetime(data[i,0]) + increment_timedelta), data[i,1] + ((k) * increment)])
                    insert_indxs.append(i+1)
                    k +=1
    if len(insert_vals) != 0:
        data = np.insert(data,insert_indxs,np.array(insert_vals),0)
    return data

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
                    break
            if flag:
                return_data[index,0] = data[i,0]
                return_data[index,1] = data[i,1]
                index +=1

    # Remove excess data 
    return return_data[~np.all(return_data == 0, axis=1)]


def remove_5_min_intervals(data: np.ndarray, is_julian: bool = False , convert_to_array: bool = False):
    """
    Remove all samples that are not in 15 minute intervals and correct 'off by one second' errors 

    data:             timeseries
    is_julian:        are the timestamps in julian time? 
    convert_to_array: should the return type be a list of numpy arrays or a 2D numpy array? 

    return:           cleansed timeseries -> list of numpy arrays or a 2D numpy array depending
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
    
    if convert_to_array:
        return_data = np.array(return_data)
        
    return return_data

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

def flip_timeseries(data : np.ndarray) -> np.ndarray:
    """
    Flip the timeseries given; done in place
    
    data:   timeseries 

    return: flipped timeseries
    """
    for i in range(data.shape[0]):
        if data[i, 1] < 0:
            data[i, 1] = 0
    max_raw = max(data[:, 1])

    for i in range(data.shape[0]):
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

def delete_missing_data_events(events: list[np.ndarray], missing_file_path: str) -> list[np.ndarray]:
    """
    Remove any events that intersect with data that should not be considered.
    
    events:            events to filter
    missing_file_path: file path of file containing dates that should not be considered
    
    return:             filtered dates
    """
    with open(missing_file_path,newline='') as file:
        # Extract missing data dates
        reader = csv.reader(file, delimiter = ',')
        time_list = []
        for row in reader:
            time_list.append([datetime_to_julian(datetime.datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S')),
                              datetime_to_julian(datetime.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S'))])
        file.close()
        
        # Filter events
        filtered_events=[]
        for event in events:
            flag = True
            start = event[0][0]
            end = event[event.shape[0]-1][0]

            for pair in time_list:
                if (pair[0] < start and start < pair[1]) or (pair[0] < end and end < pair[1]):
                    flag = False
                    break
            if flag:
                filtered_events.append(event)
                
        # Convert filtered back to ndarray
        return filtered_events

def align_stage_to_fDOM(fDOM_data : np.ndarray, stage_data : np.ndarray) -> np.ndarray:
    """
    Remove datapoints in stage that do not 
    have corresponding fDOM timestamps

    fDOM_data:  fDOM timeseries
    stage_data: stage timeseries

    return:     stage timeseries aligned to fDOM   
    """
    # Convert from julian time to datetime objects
    fDOM_data = [[julian_to_datetime(entry[0]), entry[1]] for entry in fDOM_data]
    stage_data = [[julian_to_datetime(entry[0]), entry[1]] for entry in stage_data]
        
    # Adjust data that is off by one second
    fDOM_data = correct_one_second_error(fDOM_data)
    stage_data = correct_one_second_error(stage_data)

    take_indices = []
    for i in range(len(fDOM_data)):
        j=i
        while fDOM_data[i][0] != stage_data[j][0]:
            j+=1
        take_indices.append(j)
    stage_data = np.array([[datetime_to_julian(row[0]), row[1]] for row in stage_data])
    return(np.take(stage_data, take_indices, 0))

def stage_rises_to_data(signals : np.ndarray, data : np.ndarray) -> np.ndarray:
    """
    Givin stage rise signals, correlate these signals to the original data to 
    extract edges (timestamp, value)
    
    signals: signal for each data point in timeseries: 1 == rising edge, else 0 
    data:    stage data timeseries 
    
    return: 2d array of all rising edge data points
    
    """
    take_indices = []
    for i in range(len(signals)):
        if signals[i] == 1:
            take_indices.append(i)
    return np.take(data,take_indices, 0)

def merge_additional_data(data : list, add_data : list, add_flag : str) -> list:
    """
    Merge additional data to timeseries data that already has flags

    data:     timeseries in form of list with flags to merge additional flags into 
    add_data: list of datapoint timestamps at which to add new flag
    add_flag: flag to add for new data

    return:   merged data with new flags
    """
    index = 0 
    for i in range(len(data)):
        if (index < len(add_data) and data[i][0] == add_data[index]):
            data[i][2] = add_flag
            index+=1
    return data

def delete_missing_data_peaks(data, peaks, props, missing_file_path):
    """ 
    Delete peaks that occur during time periods designated as "missing data"
    
    data:              timeseries that peaks occured in
    peaks:             indices of peaks detected
    props:             properties associated with each peak
    missing_file_path: file path of missing date ranges
    return:            filtered peaks and props
    """
    with open(missing_file_path,newline='') as file:
        reader = csv.reader(file, delimiter = ',')
        time_list = []
        for row in reader:
            time_list.append([datetime_to_julian(datetime.datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S')),
                              datetime_to_julian(datetime.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S'))])
            
        # Identify and remove violating peaks 
        keep_indices = list(np.linspace(0,peaks.shape[0]-1,peaks.shape[0]))
        for i,idx in enumerate(peaks): 
            time = data[idx,0]
            for row in time_list: 
                if time >= row[0] and time <= row[1]:
                    keep_indices.remove(i)
                    break
        
        peaks = np.take(peaks,keep_indices,0)  
        
        # Remove properties for violating peaks
        for key in props:
            props[key] = np.take(props[key], keep_indices,0)
        
        return peaks, props