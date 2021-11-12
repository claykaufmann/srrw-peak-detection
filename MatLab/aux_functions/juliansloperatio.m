%Calculate the slope ratio of a given event
%   [ratio] = juliansloperatio(event) 
%   
%   ratio: ratio of left side of event over right side of event
%   event: the given event
function ratio = juliansloperatio(event)

    metadata = computehydroITC(event);
    peak_val = max(event(:, 2));
    start_val = event(1, 2);
    end_val = event(end,2);
    
    rising_slope =  (peak_val - start_val)/ (metadata.peak - metadata.start);
    falling_slope =  (peak_val - end_val)/ (metadata.end - metadata.peak);

    ratio = rising_slope/falling_slope;
end