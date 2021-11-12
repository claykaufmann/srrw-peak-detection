%Flag runoffs extracted using HydRun for various suspicious peaks
%   [flagged_runoffs, nRunoffEvent] = flag_runoffs(data, PKThreshold, ReRa)
%   
%   flagged_runoffs: a matrix where entries in column 1 are arrays of
%   [time, value] pairs, and entries in column 2 are flags such as 'tilted'
%   or 'valid'
%   nRunoffEvent: the number of events
%
%   the computation is performed on runoffs extracted using HydRun's
%   'extractrunoff' function
function [flagged_runoffs, nRunoffEvent] = flag_runoffs(data, PKThreshold, ReRa, slope_start, slope_end,smoothing_passes, slope_ratio, stage_events, nstage_events, turb_events, nturb_events)
    hour = 0.0001140771161305; % an hour in julian year time
    [flagged_runoffs, nRunoffEvent] = extractrunoff(data, PKThreshold, ReRa, slope_start, slope_end, smoothing_passes); 
    
    basewidth_threshold = 25;
    for i = 1:nRunoffEvent
        flag = 'valid';
        turb_check = 'invalid';
        stage_check = 'invalid';
        event = flagged_runoffs{i};
        fDOM_start = flagged_runoffs{i}(1);

        if size(event, 1) < basewidth_threshold
            flag = 'invalid';  % singe data point
        elseif juliansloperatio(event) < slope_ratio
            flag = 'tilted';  % slope ratio < 
        else
            % turb
            for j = 1:nturb_events
                turb_start = turb_events{j}(1);
                if turb_start > fDOM_start
                    break
                elseif fDOM_start - turb_start < hour
                    turb_check = 'valid';
                    break
                end
            end

            % stage
            for j = 1:nstage_events
                stage_start = stage_events{j}(1);
                if stage_start > fDOM_start
                    break
                elseif fDOM_start - stage_start < hour
                    stage_check = 'valid';
                    break
                end
            end
        end
        
        if strcmp(stage_check, 'valid') && strcmp(turb_check, 'valid')
            flag = 'valid';
        else
            flag = 'p3violation';
        end

        flagged_runoffs{i,2} = flag;
    end
 

end