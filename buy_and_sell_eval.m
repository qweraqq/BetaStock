filename = 'F:\BetaStock\y_test_mse.txt';
delimiter = '';
formatSpec = '%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);
y_real = [dataArray{1:end-1}];

filename = 'F:\BetaStock\y_predict_wmse_trans.txt';
delimiter = '';
formatSpec = '%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);
y_pred = [dataArray{1:end-1}];
%% 清除临时变量
clearvars filename delimiter formatSpec fileID dataArray ans;

baseline = zeros(length(y_real)+1,1);
mymodel = zeros(length(y_real)+1,1);

baseline(1,1) = 100;
mymodel(1,1) = 100;

for t=2:length(y_real)+1
    baseline(t,1) = baseline(t-1,1)*(1+y_real(t-1,1)/100);
    if y_pred(t-1,1)>0
        mymodel(t,1) = mymodel(t-1,1)*(1+y_real(t-1,1)/100);
    elseif y_pred(t-1,1)<=0
        mymodel(t,1) = mymodel(t-1,1);
    end
end
