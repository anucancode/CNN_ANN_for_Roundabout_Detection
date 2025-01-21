clear; 
close all;
clc

%% Parameters and Folders

imageSize = 128;
imageFolder = "C:\Users\sjoer\Documents\BDSD\Minor Project\Training\RealDataSet_128";
labelFolder = "C:\Users\sjoer\Documents\BDSD\Minor Project\Training\labels\RABInfoRealDataSet.txt";

exclutionArray = [398 399 401 402 404 405 406 407 408 410 412 413 414 415 416 418 419 421 425 427 429];

%% Create DataSet
imageFileNames = dir(fullfile(imageFolder));
imageFileNames = imageFileNames(3:end,:);
imageFileNames(exclutionArray, :) = [];

opts = delimitedTextImportOptions("NumVariables", 4);

% Specify range and delimiter
opts.DataLines = [1, Inf];
opts.Delimiter = [",", "[", "]"];

% Specify column names and types
opts.VariableNames = ["Var1", "VarName2", "VarName3", "VarName4"];
opts.SelectedVariableNames = ["VarName2", "VarName3", "VarName4"];
opts.VariableTypes = ["string", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");

% Import the data
labels = readtable(labelFolder, opts);
labels = table2array(labels);
labels = labels*(imageSize/512); % LABELS ARE CHANGED AUTOMATICAlLY TO SIZE

labels(exclutionArray, :) = [];

%meanLabels = mean(labels);
%standardDev = std(labels);

%normLab = (labels - meanLabels)./standardDev;

s = length(imageFileNames);
idx = randperm(s)  ;
P1 = 0.8;
P2 = 0.9;

trainX = imageFileNames(idx(1:round(P1*s)),:); 
valX = imageFileNames(idx(round(P1*s)+1:round(P2*s)),:); 
testX = imageFileNames(idx(round(P2*s)+1:end),:) ;

trainY = labels(idx(1:round(P1*s)),:);
valY = labels(idx(round(P1*s)+1:round(P2*s)),:);
testY = labels(idx(round(P2*s)+1:end),:);


imgFilesTrain = string([length(trainX),1]);
for i = 1 : length(trainX)
    imgFilesTrain(i) = fullfile(trainX(i).folder, trainX(i).name);
end

imgFilesVal = string([length(valX),1]);
for i = 1 : length(valX)
    imgFilesVal(i) = fullfile(valX(i).folder, valX(i).name);
end

imgFilesTest = string([length(testX),1]);
for i = 1 : length(testX)
    imgFilesTest(i) = fullfile(testX(i).folder, testX(i).name);
end

imdsTrain = imageDatastore(imgFilesTrain);
labdsTrain = arrayDatastore(trainY, 'IterationDimension', 1);

imdsVal = imageDatastore(imgFilesVal);
labdsVal = arrayDatastore(valY, 'IterationDimension', 1);

imdsTest = imageDatastore(imgFilesTest);
labdsTest = arrayDatastore(testY, 'IterationDimension', 1);

dataSetTrain = combine(imdsTrain,labdsTrain);
dataSetVal = combine(imdsVal,labdsVal);
dataSetTest = combine(imdsTest,labdsTest);

%% Define Layers

error = inf;

for i = 1 : 100

layers = [
    imageInputLayer([imageSize imageSize 1]);
    convolution2dLayer(3,64,'padding','same');
    reluLayer();
    convolution2dLayer(3,64,'padding','same');
    reluLayer();
    maxPooling2dLayer(2,'padding','same');
    convolution2dLayer(3,128,'padding','same');
    reluLayer();
    convolution2dLayer(3,128,'padding','same');
    reluLayer;
    maxPooling2dLayer(2,'padding','same');
    % convolution2dLayer(3,256,'padding','same');
    % reluLayer();
    % convolution2dLayer(3,256,'padding','same');
    % reluLayer;
    % maxPooling2dLayer(2,'padding','same');
    convolution2dLayer(3,imageSize,'padding','same');
    reluLayer();
    convolution2dLayer(3,imageSize,'padding','same');
    reluLayer();
    convolution2dLayer(3,4,'padding','same');
    reluLayer();
    convolution2dLayer(3,4,'padding','same');
    reluLayer();
    flattenLayer();
    fullyConnectedLayer(imageSize/2)
    reluLayer();
    fullyConnectedLayer(16)
    reluLayer();
    fullyConnectedLayer(3)
    regressionLayer];


%% Define training options


opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 25, ...
    'Verbose', true);


%% Train

disp(append("start: ", string(datetime)))

[net, info] = trainNetwork(dataSetTrain, layers, opts);

save("NetworkCNN.mat","info","net")

%% Validate

prediction = predict(net, dataSetVal);
newError = sqrt(mean((valY(:)-prediction(:)).^2));
disp(append("NEW MSE: ",string(newError)," - OLD ERROR: ",string(error)))

if(error > newError)
    fprintf("NEW NETWORK SAVED\n\n")
    save("NetworkCNN.mat","info","net")
    error = newError;
else
    fprintf("No improvement\n\n")
end

end
