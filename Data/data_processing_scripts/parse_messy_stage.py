# Stage data formatting is different than that of fDOM and turbidity
import csv
import sys
import datetime
sys.path.insert(1,'../')
from Tools.data_processing import datetime_to_julian

def parse_stage_data(args):
    """
    Parse messy stage data (as we recieve it) to useful format 
    
    args: list containing:
            - file path to messy data 
            - file path where parsed data will be written out to 
            - desired format of timestamps: "julian" or "standard" datetime
    """
    data = []
    # Read in data 
    with open(args[0], newline='') as in_file:
        reader = csv.reader(in_file, delimiter ='\t')
        next(reader) # drop headers
        for row in reader:
            data.append(row)
        in_file.close()
        
    valid_min = [0,15,30,45]
    time_delta = datetime.timedelta(seconds = 1)
    # Write out data 
    with open(args[1], 'w',newline = '') as out_file:
        writer = csv.writer(out_file, delimiter = ',')
        for row in data:  
            date = datetime.datetime.strptime(row[0],'%m/%d/%Y %H:%M:%S')
            
            # Remove violating entries
            if date.second != 0: 
                date+=time_delta
            if date.minute in valid_min:
                if args[2].lower() == 'julian':
                    date = datetime_to_julian(date)
                writer.writerow([str(date),row[1]])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        args = ['/Users/zachfogg/Desktop/DB-SRRW/Data/messy_data/stage_10.1.12-1.1.19.txt',
        '/Users/zachfogg/Desktop/DB-SRRW/Data/converted_data/datetime_format/stage_10.1.12-1.1.19.txt',
        'standard']
    else: 
        args = sys.argv[1:]

    parse_stage_data(args)

 