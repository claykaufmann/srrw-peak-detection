## Documentation of files in this directory

All data in this file is uncleaned, unprocessed csv files containing data. Files contain both rellevant and irrelevant data. In general, files follow this format: 
15 lines of metadata, and then entries for the given data type (fDOM, stage, turbidity, temperature, indicated by the naming of the file) where each entry is formatted as: `[ISO 8601 UTC, Timestamp (UTC-05:00), Value, Approval Level, Grade, Qualifiers]`. Entries are almost exclusively spaced 15 minutes apart consecutively, where each value is the sensor reading for the given data type. Here, the only useful columns are: Timestamp (UTC-05:00), Value. All data is from the same geographical site: Pope Brook Tributary, North Danville VT. 

Temperature is degrees celcius. 
fDOM is measured in QSE.
Stage is measured in gage height (ft). 
Turbidity is measured in Nephelometric Turbidity Units (NTU). 

Below is listed the actually date ranges in each file, reguardless of file naming. If a file does not separate entries by 15 minute intervals, that is also listed. 

1. `Turbidity_corrected_WY2017.csv` - 10/1/2016 0:00 to 9/30/2017 23:45. 
2. `Turbidity_corrected_WY2013-2020.csv` - 10/1/2012 0:00 to 9/4/2020 11:00. 
3. `Temperature_corrected_WY2017.csv` - 10/1/2016 0:00 to 9/30/2017 23:45
4. `Stage_corrected_WY2017-2020.csv` - 9/30/2016 19:00 to 7/13/2020 12:30. 
5. `Stage_corrected_WY2017.csv` - 10/1/2016 0:00 to 10/1/2017 0:00.
6. `Stage_corrected_WY2013.csv` - 5/1/2013 0:00 to 5/31/2013 23:55. Interval is every 5 minutes instead of 15 minutes. 
7. `fDOM_raw_WY2017csv.csv` - 10/1/2016 0:00 to 10/31/2017 23:45. 
8. `fDOM_raw_WY2013-2017.csv` - 10/1/2012 0:00 to 9/4/2020 11:00. 
9. `fDOM_raw_WY2013.csv` - 5/1/2013 0:00 to 5/31/2013 23:45. 
10. `fDOM_corrected_WY2017.csv` - 10/1/2016 0:00 to 10/31/2017. 
11. `fDOM_corrected_WY2013-2020.csv` - 10/1/2012 0:00 to 9/4/2020 11:00. 
12. `fDOM_corrected_WY2013.csv` - 5/1/2013 0:00 to 5/31/2013 23:45
13. `fDOM,_water,_in_situ.FDOM_ppb_ave@01135100.20161001 (1).csv` - 10/1/2016 0:00 to 9/30/2017 23:45. This file has odd naming but is still fDOM in the specified date ranges. 
14. `Cleaning_calibration_master_T_K_FDOM_2010-present.csv(or .xls)` Sensor cleaning timestamps and comments from 4/6/2010 to 6/9/2020 
