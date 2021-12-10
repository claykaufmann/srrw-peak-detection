# Anomaly Detection and Correction of Hydrogeochemical Sensor Data(ADC)

This repository contains the codebase for work surrounding the **"Automated Cleaning of Time Series Data from the Sleepers Research River Watershed (SRRW)"** project. This project aims to develop an automatic mechanism that detects and corrects anomalies in time series data for hydrological and biogeochemical study. The time series data from the Sleepers Research River Watershed (SRRW), representing the changes of water stage (ft) and the associated temperature, turbidity, and FDOM, are used as the case study datasets.

As of this commit we are exploring rule based classifiers whose parameters are optimized through random search and small Convolutional Neural Networks, for anomaly detection.

### Current timeseries data that we consider in anomaly detection: 
- fluorescent disolved organic matter (fDOM)
- turbidity (turb)
- stage

### Current defined anomaly types:
below are the current distinct anomaly types that we have identified in our data:

#### Current anomalies recognized in fDOM: 
   - `fDOM_PLP` (fDOM Plummeting Peak) - a sharp sudden drop in an amplitude of the timeseries data, usually with a small basewidth
   - `fDOM_PP` (fDOM Phantom Peak) - an fDOM peak that does not follow the three peak ordering rule
   - `fDOM_SKP` (fDOM Skyrocketing Peak)- a sharp sudden spike in amplitude of the timeseries data, usually with a small basewidth. This excludes those cause by interference from another timeseries. 
   - `fDOM_local_fluctuation` - many small peaks in rapid succession. 
   - `fDOM_flat_plateau` - consecutive samples between a sudden rise and sudden drop during which the amplitude is “near constant"
   - `fDOM_flat_sink` - consecutive samples between a sudden drop and a sudden rise during which the amplitude is “near constant”
   - `fDOM_sensor_drift` - not yet detected
   
#### Current anomalies recognized in turbidity:
   - `turb_local_fluctuation` - many small peaks in rapid succession. 
   - `turb_PP` (turbidity Phantom Peak) - an turb peak that does not follow the three peak ordering rule
   - `turb_sensor_drift` - not yet detected
   - `turb_SKP ` (turbidity Skyrocketing Peak) - a sharp sudden spike in amplitude of the timeseries data, usually with a small basewidth, with no corresponding fDOM peak. 
   - `turb_flat_plateau` - consecutive samples between a sudden rise and sudden drop during which the amplitude is “near constant"
   - `turb_flat_sink` - consecutive samples between a sudden drop and a sudden rise during which the amplitude is “near constant”

## Explanation of Repository Contents: 

## Anomaly_Detection/ 
- contains **.py** and **.ipynb** files associated with detecting predefined anomaly types. The files within are continually updating and evolving so descriptions will be generated when they are in a more permenant state. 

## Anomaly_Correction/
- contains **.py** and **.ipynb** files associated with correcting predefined anomaly types. The files here are outdated

## Data/
- contains all data used and generated in this project: raw timeseries, manually annoated data, detected anomalies, ect...

#### 1. **Data/temp_plotting/** 
- contains **.csv** files representing data to be plotted to TrainSet, whether that be for labeling or demonstration purposes

#### 2. **Data/converted_data/** 
- contains processed **.csv** files with the two columns `[Julian_time, value]` or `[Datetime, value]` generated from the messy_data files and also two **.csv** files containing the time log `[date, time]` of sensor cleaning. It contains both raw and corrected data (raw meaning the values are untouched, corrected meaning that the values have been hand corrected by a domain scientist). 

#### 3. **Data/data_processing_scripts/**
- scripts to convert "messy data" (as recieved from domain scientists) to "converted" data that is in a usable format. 

#### 4. **Data/imgs/**
- images that were captured throughout the project that were deemed worth saving

#### 5. **Data/manual_annotating_data/**
- labeled data, formatted as: 1. as downloaded from TrainSet. 2. converted into "ground_truth" files that will be used in training

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

