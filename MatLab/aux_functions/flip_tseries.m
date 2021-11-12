%Flip a timeseries upside down
%   [flip] = flip_tseries(flip)
%   
%   flip: the series to flip, the flip is done in place so this is the
%   output as well. The series in the form of an array of [time, value]
%   pairs
function flip = flip_tseries(flip)

    for i = 1:length(flip)
        if flip(i,2) < 0
            flip(i,2) = 0;
        end
    end
    max_raw = max(flip(:, 2));
    
    for i = 1:length(flip)
        flip(i,2) = max_raw - flip(i,2);
    end
    

end