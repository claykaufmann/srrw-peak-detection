function interp = interp_tseries(interp, pks,locs,w,p)
    for i = 1:length(locs)
        try
            half = floor(w(i)/2);
            half = 20;
            st_ind = locs(i)-half;
            end_ind = locs(i)+half;
            st_val = interp(st_ind,2);
            end_val = interp(end_ind,2);
            len = length(interp(st_ind:end_ind, 2));
            interp(st_ind:end_ind, 2)=linspace(st_val, end_val, len);
        catch
        end
    end
end