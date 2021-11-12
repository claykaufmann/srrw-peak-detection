%Interpolate a set of flagged events
%   [interpolated_raw] = interp_flagged_events(flagged_events, n_events, raw)
%   
%   interpolated_raw: the raw array after being interpolated
%
%   flagged_events: array of flagged events with flags
%   n_events: number of flagged events
%   raw: the entire timeseries, including entries not flagged as events
function interpolated_raw = interp_flagged_events(flagged_events, n_events, raw)
    for i = 1:n_events
        runoff = flagged_events{i};
        flags = flagged_events{i, 2};
        
        % This appears to only interpolate an event that has violated the
        % basewidth threshold (sharp peak).  
%         if strcmp(flag,'invalid') 
        if flags(1) || flags(2) % This will interpolate over any basewidth violating event: single point of sharp peak. 
            st = runoff(1, 1); 
            disp(i);
            ed = runoff(end, 1);

            [~, ipeak] = max(runoff(:, 2)); 

            idx = raw(:, 1) >= st & raw(:, 1) <= ed;
          
            st_val = runoff(1, 2);
            end_val = runoff(length(runoff), 2);
            len = length(runoff);
            
            runoff(:, 2) = linspace(st_val, end_val, len);
            raw(idx, :) = runoff;

        end
      
        flagged_events{i} = runoff;
    end
    interpolated_raw = raw;
end