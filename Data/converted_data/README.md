## Documentation of files in this directory:

Files contain the specified data type in the specified data ranges. The data is stored in a matlab array in form `[value, timestamp]`; matlab arrays can easily be loaded at numpy arrays in python. The timestamp is recorded in julian time. Example: `stage_10.1.12-7.13.20.mat` contains conseuctive `[stage_value, julian_time_of_reading]`entries from 10/1/12 to 7/13/20 where each entry is a recorded sensor reading 15 minutes after the last. All data is collected from the same geographical site: Pope Brook Tribuatary, North Danville VT. 

Files containing '_corrected_' are corrected data. Corrected data is a timeseries that have had some types of anomalies corrected. Currently only 'Sharp / Plummetting/ Skyrocketing Peaks' and 'Single Point Peaks' are corrected: corrected by interpolation. Files containing '_raw_'is raw data. Raw data is as it was collected, untouched. 

Files are saved in Matlab arrays but can easily be read in as numpy python arrays. 

fDOM_cleaning_*.csv contain dates of fDOM sensor cleanings. 
