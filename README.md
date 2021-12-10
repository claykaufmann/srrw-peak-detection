# Anomaly Detection and Correction of Hydrogeochemical Sensor Data(ADC)

This repository contains the codebase for work surrounding the **"Automated Cleaning of Time Series Data from the Sleepers Research River Watershed (SRRW)"** project. This project aims to develop an automatic mechanism that detects and corrects anomalies in time series data for hydrological and biogeochemical study. The time series data from the Sleepers Research River Watershed (SRRW), representing the changes of water stage (ft) and the associated temperature, turbidity, and FDOM, are used as the case study datasets.

As of this commit we are exploring rule based classifiers whose parameters are optimized through random search. However, we are soon going to explore deep learning methods: RNN, LSTM, GRU ...

### Current defined anomaly types:
below are the current distinct anomaly types that we have identified in our data:

#### Current anomalies detected in fDOM: 
   - `normal_peak_fDOM` - although not an anomaly, normal peaks must also be detected
   - `plummeting_peak_fDOM` - a sharp sudden drop in an amplitude of the timeseries data, usually with a small basewidth
   - `out_of_place_peak_fDOM` - an fDOM peak that does not follow the three peak ordering rule
   - `skyrocketing_peak_fDOM` - a sharp sudden spike in amplitude of the timeseries data, usually with a small basewidth. This excludes those cause by interference from another timeseries. 
   - `local_fluctuation_fDOM` - many small peaks in rapid succession. 
   - `flat_plateau_fDOM` - consecutive samples between a sudden rise and sudden drop during which the amplitude is “near constant"
   - `flat_sink_fDOM` - consecutive samples between a sudden drop and a sudden during which the amplitude is “near constant”
   - `sensor_drift_fDOM` - not yet detected
   
#### Current anomalies detected in turbidity
   - `local_fluctuation_turb` - many small peaks in rapid succession. 
   - `out_of_place_peak_turb` - an turb peak that does not follow the three peak ordering rule
   - `normal_peak_turb` - although not an anomaly, normal peaks must also be detected
   - `sensor_drift_turb` - not yet detected
   - `skyrocketing_peak_turb` - a sharp sudden spike in amplitude of the timeseries data, usually with a small basewidth, with no corresponding fDOM peak. 
   - `flat_plateau_turb` - consecutive samples between a sudden rise and sudden drop during which the amplitude is “near constant"
   - `flat_sink_turb` - consecutive samples between a sudden drop and a sudden during which the amplitude is “near constant”

## Explanation of Repository Contents: 

## Anomaly_Detection/ 
- contains **.py** and **.ipynb** files associated with detecting predefined anomaly types. The files within are continually updating and evolving so descriptions will be generated when they are in a more permenant state. 

## Anomaly_Correction/
- contains **.py** and **.ipynb** files associated with correcting predefined anomaly types. The files here are outdated

## Data/
- contains all data used and generated in this project: raw timeseries, manually annoated data, detected anomalies, ect...

#### 1. **Data/anomaly_data/** 
- contains **.csv** files representing current fDOM and turbidity anomalies detected in the data. Each **.csv** files is in the form `[start_time, end_time, peak_time, anomaly_type]` (unless otherwise specified) where the timestamps are in julian time, and the **anomaly_type** is a string representing one of our current defined anomaly types. 

#### 2. **Data/converted_data/** 
- contains processed **.csv** files with the two columns `[Julian_time, value]` or `[Datetime, value]` generated from the messy_data files and also two **.csv** files containing the time log `[date, time]` of sensor cleaning. It contains both raw and corrected data (raw meaning the values are untouched, corrected meaning that the values have been hand corrected by a domain scientist). 

#### 3. **Data/data_parsing_scripts/**
- scripts to convert "messy data" (as recieved from domain scientists) to "converted" data that is in a usable format. 

#### 4. **Data/imgs/**
- images that were captured throughout the project that were deemed worth saving

#### 5. **Data/manual_annotating_data/**
- data involved in manual data annotation (creating of training and testing data): data to be labeled, annotated data, and annotated data that has been processed to extract relevant information

#### 6. **Data/messy_data/** 
- contains the time series data as recieved. It usually comes as a csv file with some extraneous columns, so it is pared down to just the value and time columns. The time column is then converted to julian time and datetime for ease of computation. In addition, there are field note files `[date, time, comment]`, containing the sensor cleaning logs.

#### 7. **Data/misc/**
- contains miscallaneous data 

## Data_Annotating_Scripts/
- contains scripts associated with processing manually annotated data to extract only relevant information 

## MatLab/
- contains outdated, unmaintained MatLab files; they are preserved for reference. The codebase for this project was originally written in MatLab due to a previously used tool ([HydRun](https://github.com/weigangtang/HydRun)) being written in MatLab. The codebase has since then been migrated to exclusively Python. 

## Tools/ 

- contains python modules with many helper functions created to assist in anomaly detection and correction. These functions are documented in their respective files

## Assisting Tools

1. [TrainSet](https://github.com/geocene/trainset) is a tool for efficiently manually annotating data timeseries. We use it to manually annotate peaks and anomalies to later be used as train/test sets. 

