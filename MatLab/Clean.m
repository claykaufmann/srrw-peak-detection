%% START HYPERPARAMS

year = "2017";
slope_ratio = 1;

% for fDOM event separation
fDOM_PKThreshold = 3; % peak threshold for runoff events
fDOM_ReRa = 0.999; % return ratio

% for removal of negative fDOM spikes
min_flipped_height = 103;

% for stage event separation
stage_PKThreshold = 0.02;
stage_ReRa = 0.6;

% for stage event separation
turb_PKThreshold = 15;
turb_ReRa = 0.6;

%% START LOADING DATA

% add function folder to MATLAB search path
addpath('HydRun/HydRun_functions'); 
addpath('aux_functions');

% loading the matlab data: time series of streamflow and precip (rainfall)
load('converted_data/fDOM_raw_WY2013-2020.mat', 'fDOM_raw'); 
load('converted_data/stage_corrected_WY2017-2020.mat', 'stage');
load('converted_data/turbidity_corrected_WY2013-2020.mat', 'turb');
load('converted_data/fDOM_corrected_WY2013-2020', 'fDOM_corrected');
fDOM_corrected(:,1) = fDOM_corrected(:,1) + 0.224 * 1;

%% START PROCESSING/ DESPIKING

echo on

% this extracts events for stage
[stage_events, nstage_events] = extractrunoff(stage, stage_PKThreshold, stage_ReRa, 0.001, 0.0001, 4);

% this extracts events for turbidity
[turb_events, nturb_events] = extractrunoff(turb, turb_PKThreshold, turb_ReRa, 0.001, 0.0001, 4);

% extract runoff events from fDOM
[fDOM_events, nfDOM_events] = flag_runoffs(fDOM_raw, fDOM_PKThreshold, fDOM_ReRa, slope_ratio, stage_events, nstage_events, turb_events, nturb_events); 
fDOM_despiked = interp_flagged_events(fDOM_events, nfDOM_events, fDOM_raw);

% this flips the fDOM upside down to get rid of negative spikes
flipped_fDOM = flip_tseries(fDOM_despiked);
[pks,locs,w,p] = findpeaks(flipped_fDOM(:,2),'MinPeakHeight',min_flipped_height);
fDOM_despiked = interp_tseries(fDOM_despiked, pks,locs,w,p); 

%% START PLOTTING

% first subplot is raw vs despiked fDOM, with some types of spikes flagged
axs1 = subplot(3, 1, 1);
hold on;

title("Corrected vs Despiked fDOM")
% plot(fDOM_corrected(:,1), fDOM_corrected(:,2), '-c');
plot(datetime(fDOM_raw(:,1),'convertfrom','juliandate'), fDOM_raw(:,2), '-', 'color', [1/178 1/100 1/5]);

% smoothed = smoothdata(fDOM_despiked(:,2),'movmedian',3);
plot(datetime(fDOM_despiked(:,1),'convertfrom','juliandate'), fDOM_despiked(:,2), 'color', [0.95 0.5 0.2]);
legend('Raw','Despiked');

% instead of removing some peaks (like tilted peaks), flag them here
for i = 1:nfDOM_events
    runoff = fDOM_events{i};
    flag = fDOM_events{i, 2};

    if strcmp(flag,'tilted')
        [peak, ipeak] = max(runoff(:, 2));
        plot(datetime(runoff(ipeak,1),'convertfrom','juliandate'), runoff(ipeak,2), '*b');
    end

end

% plot the HydRun event extracted stage for comparison
axs2 = subplot(3, 1, 2);

title('Stage ft')
plotrunoffevent(stage_events, stage); 

% plot the turbidity for comparison
axs3 = subplot(3, 1, 3);
title('Turb ft');
fig = plot(datetime(turb(:,1),'convertfrom','juliandate'), turb(:,2), '-',  'color', [1 0.5 0.5]);
legend('Turbidity');

linkaxes([axs1,axs2,axs3],'x');
hold off;