function data = save_timeseries_as_datetime(data,save_filename)
    % Function takes in given timeseries and saves timeseries in 
    % datetime format as mat file 
    datetime_timeseries = zeros(length(data),2);
    for i=1:length(data)
        time = datetime(data(i,1),'convertfrom','juliandate');
        str_time = sprintf("%s",time);
        datetime_timeseries(i,:) = [str_time data(i,2)];
%         datetime_timeseries(i,1) = datetime(data(i,1),'convertfrom','juliandate');
%         datetime_timeseries(i,2) = data(i,2);
    end
    save(save_filename, "datetime_timeseries"); 
end