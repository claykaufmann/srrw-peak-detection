def event_from_props_ips(peaks : np.ndarray, props : dict, data : np.ndarray) -> list[np.ndarray]:
    """
    Generate a multi-sample "event" given a peak and it's left/right interpolated positions
    
    peaks :  detected peaks indexs
    props :  detected peaks properites
    data :   timeseries that peaks are from 
    return : list of events generated from peaks
    """
#     events = []
#     prev_end = -1
#     for i,index in enumerate(peaks):
#         start = int(props['left_ips'][i])
#         end = int(props['right_ips'][i])
#         if i < len(peaks)-1:
#             next_start = int(props['left_ips'][i+1])
#             end_2 = end
#             bw = end - start
#             if end <= next_start:s
#                 end = next_start-1
#             if start >= end-1:
#                 print('\n\n', i, bw)
#                 print(next_start, start,end_2)
#             else:
#                 events.append(data[start: end + 1,:])
#         else: 
#             events.append(data[start:end+1,:])
#     return events
    events = []
    for i, index in enumerate(peaks):
        events.append(data[int(props['left_ips'][i]):int(props['right_ips'][i])+1,:])
    return events


def peak_of_event(event : np.ndarray) -> np.ndarray:
    """
    Detect the peak sample in an event
    
    event :  candidate event
    return : sample that is peak of event
    """
    max = event[0]
    for entry in event:
        if entry[1] > max[1]:
            max = entry
    return max

