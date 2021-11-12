%%
load('converted_data/fDOM_corrected_WY2017.mat', 'fDOM_corrected');
load('converted_data/fDOM_raw_WY2017.mat', 'fDOM_raw');
load('converted_data/stage_corrected_WY2017.mat', 'stage');
cleaning_datetimes = readtable('converted_data/fDOM_cleaning_2017.csv');

cleaning_julians = [];
for i = 1:size(cleaning_datetimes(:,1))
    cleaning_julians = [cleaning_julians; juliandate(cleaning_datetimes{i,1})];
end
%%
axs1 = subplot(2, 1, 1);

title("Corrected vs raw fDOM")
hold on;

plot(fDOM_raw(:,1), fDOM_raw(:,2), '-k');
plot(fDOM_corrected(:,1), fDOM_corrected(:,2), '-', 'color', [0.85, 0.58, 0.25]);
size(cleaning_julians(:,1))
for i = 1:size(cleaning_julians(:,1))
    xline(cleaning_julians(i,1), 'r');
end
legend('Original', 'Processed');

axs2 = subplot(2, 1, 2);
plot(stage(:,1), stage(:,2), '-b');
legend('Stage');

linkaxes([axs1,axs2],'x');

hold off;