function trimmed_data = trim_timeseries(data,start_time,end_time)
    % trim the given timeseries to within the specified dates: julian time
    trimmed_data = zeros(length(data),2);
    
    idx = 1; 
    for i=1:length(data)
        if data(i,1) <= end_time && data(i,1) >= start_time
            trimmed_data(idx,:) = [data(i,1) data(i,2)];
            idx= idx + 1; 
        end
    end 
    trimmed_data = trimmed_data(1:idx-1,:);
end 