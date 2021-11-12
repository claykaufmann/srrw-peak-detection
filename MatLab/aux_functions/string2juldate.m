% convert a string to a julian date (double)
function juldate = string2juldate(string)
    toRemove = isletter(string); % Detect which ones are letters
    string(toRemove) = [];     % Delete the letters

    % juldate = juliandate(datetime(string,'InputFormat','yyyy-MM-ddHH:mm:ss'));

    juldate = juliandate(datetime(string,'InputFormat','MM/d/yyyy HH:mm'));

end
