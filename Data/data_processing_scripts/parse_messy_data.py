import csv
import sys
import datetime
sys.path.insert(1,'../')
from Tools.data_processing import datetime_to_julian

def parse_messy_data(args):
    """
    Parse messy data (as we recieve it) to useful format 
    
    args: list containing:
            - file path to messy data 
            - file path where parsed data will be written out to 
            - desired format of timestamps: "julian" or "standard" datetime
            - number of header lines to skip
    """
    args[2] = args[2].lower()
    
    # Read in data
    data = []
    with open(args[0], 'r', newline='') as in_file:
        reader = csv.reader(in_file, delimiter = ',')
        
        # Skip first 15 lines
        for i in range(int(args[3])): next(reader)
        
        # Read data
        for row in reader: 
            data.append([row[1], float(row[2])])
            
        in_file.close()
        
    valid_min = [0,15,30,45]
    time_delta = datetime.timedelta(seconds = 1)
    filtered_data = []
    for i in range(len(data)):
        date = datetime.datetime.strptime(data[i][0], '%Y-%m-%d %H:%M:%S')
        
        # Remove violating entries
        if date.second != 0: 
            date+=time_delta
            
        if date.minute in valid_min:
            # Convert time to desired format
            if args[2] == "julian":
                date = datetime_to_julian(date)
            filtered_data.append([str(date),data[i][1]]),
        
    # Write out to file 
    with open(args[1], 'w', newline = '') as out_file:
        writer = csv.writer(out_file, delimiter = ',')
        for row in filtered_data: 
            writer.writerow(row)
        out_file.close()
        

if __name__ == "__main__":
    if len(sys.argv) != 5:
        args = ['/Users/zachfogg/Desktop/DB-SRRW/Data/messy_data/turbidity_raw_10.1.2011_9.4.2020.csv',
        '/Users/zachfogg/Desktop/DB-SRRW/Data/converted_data/julian_format/turbidity_raw_10.1.2011_9.4.2020.csv',
        'julian',
        15]
    else: 
        args = sys.argv[1:]
    parse_messy_data(args)

 