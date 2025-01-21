clear; 
close all;
clc

%% Parameters and Folders

rng(9022002)

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
P = 0.8;

trainX = imageFileNames(idx(1:round(P*s)),:) ; 
testX = imageFileNames(idx(round(P*s)+1:end),:) ;

trainY = labels(idx(1:round(P*s)),:);
testY = labels(idx(round(P*s)+1:end),:);

imgFilesTrain = string([length(trainX),1]);
for i = 1 : length(trainX)
    imgFilesTrain(i) = fullfile(trainX(i).folder, trainX(i).name);
end

imgFilesTest = string([length(testX),1]);
for i = 1 : length(testX)
    imgFilesTest(i) = fullfile(testX(i).folder, testX(i).name);
end

imdsTrain = imageDatastore(imgFilesTrain);
labdsTrain = arrayDatastore(trainY, 'IterationDimension', 1);

imdsTest = imageDatastore(imgFilesTest);
labdsTest = arrayDatastore(testY, 'IterationDimension', 1);

dataSetTrain = combine(imdsTrain,labdsTrain);
dataSetTest = combine(imdsTest,labdsTest);

%% Define Layers

layers = [
    imageInputLayer([imageSize imageSize 3]);
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
    convolution2dLayer(3,256,'padding','same');
    reluLayer();
    convolution2dLayer(3,256,'padding','same');
    reluLayer;
    maxPooling2dLayer(2,'padding','same');
    convolution2dLayer(3,imageSize,'padding','same');
    reluLayer();
    convolution2dLayer(3,imageSize,'padding','same');
    reluLayer();
    convolution2dLayer(3,4,'padding','same');
    reluLayer();
    convolution2dLayer(3,4,'padding','same');
    reluLayer();
    flattenLayer();
    fullyConnectedLayer(imageSize*3)
    reluLayer();
    fullyConnectedLayer(imageSize)
    reluLayer();
    fullyConnectedLayer(imageSize/2)
    reluLayer();
    fullyConnectedLayer(16)
    reluLayer();
    fullyConnectedLayer(3)
    regressionLayer];


%% Define training options


opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-14, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 5000, ...
    'MiniBatchSize', 50, ...
    'Verbose', true, ...
    'CheckpointPath', pwd);

%% Load Previous Weights (if possible)
networkFile = 'NetworkCNN.mat';

if isfile(networkFile)
    disp("PRE-Trained network found.")
    load(networkFile)
    layers = net.Layers;
else
    disp("Training new network")
end

%% Train

disp(append("start: ", string(datetime)))

[net, info] = trainNetwork(dataSetTrain, layers, opts);

save("NetworkCNN.mat","info","net")

%% Test

prediction = predict(net, dataSetTest);

%% Visualize

for i = 1 : length(imdsTest.Files)
    disp(i)
    img = imread(imgFilesTest(i));
    img = insertShape(img,"circle",prediction(i,:),LineWidth=2);
    
    fig = figure(1);
    imshow(img)

    disp("PAUSED - click enter")
    pause
end

return

%% Visualize Train

for i = 390 : length(imageFileNames)
    disp(i)
    img = imread(fullfile(imageFileNames(i).folder, imageFileNames(i).name));
    img = insertShape(img,"circle",labels(i,:),LineWidth=2);
    

    fig = figure(1);
    imshow(img)

    disp("PAUSED - click enter")
    pause
    clc
end