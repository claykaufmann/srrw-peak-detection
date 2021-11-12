% a couple different options for parsing the data, usually the first two
% cells work with the input format
filename = 'messy_data/fDOM_corrected_2013_2017.csv';
% dirty_table = readtable(filename, 'Format', '%{MM/dd/yy HH:mm}D');
dirty_table = readtable(filename);

% dirty_table= removevars(dirty_table,{'ISO8601UTC', 'ApprovalLevel', 'Grade', 'Qualifiers'});

dirty_table= removevars(dirty_table,{'x_FDOM', 'Var4', 'Var5', 'Var6'});

dirty_table((1:15),:) = [];

% dates = char(table2array(dirty_table(1:10, 91)))
% vals = (table2array(dirty_table(1:10, 2)))

%%

dirty_table{:, 1} = datetime(dirty_table{:, 1});
%%
stage = zeros(size(dirty_table, 1),2);

for i=1:(size(dirty_table, 1))
    stage(i, 1) = juliandate(datetime((dirty_table{i, 1})));
    stage(i, 2) = dirty_table{i, 2};
end

%%
% echo on
data = table2array(dirty_table);

for i=1:length(data)
    
    data{i, 1} = string2juldate(data(i, 1));
    data{i, 2} = str2double(data{i, 2});
end

%%

fDOM_corrected=data

%%
data = cell2mat(data);
fDOM_corrected = data;