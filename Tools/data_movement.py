"""
This module contains the following functions: 

- read_in_timeseries
- read_in_preprocessed_timeseries
- write_data_to_trainset
- write_detected_to_trainset

"""

import csv
import datetime
import sys
import numpy as np
#sys.path.append('~/Projects/Hydrogeochemical_ADC/Tools/') # Jupyter-lab has a bug that makes this required
import Tools.data_processing as dp
import numpy as np

def read_in_timeseries(in_file : str, is_julian : bool = True) -> list[list]:
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

def read_in_preprocessed_timeseries(in_file : str) -> np.ndarray:
    """
    Read timseries in from csv, trim to correct length, delete missing data ranges, 
    interpolate small missing data intervals 

    in_file: file path 
    return:  preprocessed data
    """
    data = np.array(read_in_timeseries(in_file, True))
    start_time = 2456042.5
    end_time = 2458484.5

    data = dp.trim_timeseries(data, start_time, end_time)

    data = dp.delete_from_timeseries(data, '../Data/misc/delete_date_ranges.txt')

    data = dp.interpolate_missing_intervals(data)

    return data

# Specifically for plotting annotated data to the 'Trainset' website plotting software 
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
        fDOM_data = dp.add_flags(fDOM_data)
        stage_data = dp.add_flags(stage_data)
        turb_data = dp.add_flags(turb_data)
    
    # Convert from julian time to datetime objects
    if is_julian_time:
        fDOM_data = [[dp.julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in fDOM_data]
        stage_data = [[dp.julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in stage_data]
        turb_data = [[dp.julian_to_datetime(entry[0]), entry[1], entry[2]] for entry in turb_data]
        
    # Adjust data that is off by one second
    fDOM_data = dp.correct_one_second_error(fDOM_data)
    stage_data = dp.correct_one_second_error(stage_data)
    turb_data = dp.correct_one_second_error(turb_data)
    
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
                
                writer.writerow(['fDOM', fDOM_time, f'{fDOM_data[i][1]:.5f}', fDOM_data[i][2]])
                writer.writerow(['Stage', stage_time, f'{stage_data[i][1]:.5f}', stage_data[i][2]])
                writer.writerow(['Turbidity',turb_time, f'{turb_data[i][1]:.5f}', turb_data[i][2]])
    outfile.close()

# Specifically for plotting annotated data to the 'Trainset' website plotting software 
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
    fDOM_events = dp.collapse_events(fDOM_events)
    stage_events = dp.collapse_events(stage_events)
    turb_events = dp.collapse_events(turb_events)
    
    # Merge each timeseries with events
    fDOM_merged = dp.merge_data(fDOM_data, fDOM_events, flags['fDOM_flag'], flags['default_fDOM_flag'])
    stage_merged = dp.merge_data(stage_data, stage_events, flags['stage_flag'], flags['default_stage_flag'])
    turb_merged = dp.merge_data(turb_data, turb_events, flags['turb_flag'], flags['default_turb_flag'])
   
    write_data_to_trainset(fDOM_merged, stage_merged, turb_merged, out_file, True, True, data_start, data_end)
