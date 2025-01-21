clear; 
close all;
clc

%% Parameters and Folders

rng(9022002)

imageSize = 224;
imageFolder = "C:\Users\sjoer\Documents\BDSD\Minor Project\Training\RealDataSet_224";
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

trainY = labels(idx(1:round(P*s)),3);
testY = labels(idx(round(P*s)+1:end),3);
testYfull = labels(idx(round(P*s)+1:end),:);

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

params = load("C:\Users\sjoer\Documents\BDSD\Minor Project\Training\CNNMatlab\params_2024_06_16__12_08_58.mat");

layers = [
    imageInputLayer([224 224 3],"Name","input","Mean",params.input.Mean)
    convolution2dLayer([3 3],64,"Name","conv1_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv1_1.Bias,"Weights",params.conv1_1.Weights)
    reluLayer("Name","relu1_1")
    convolution2dLayer([3 3],64,"Name","conv1_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv1_2.Bias,"Weights",params.conv1_2.Weights)
    reluLayer("Name","relu1_2")
    maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv2_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv2_1.Bias,"Weights",params.conv2_1.Weights)
    reluLayer("Name","relu2_1")
    convolution2dLayer([3 3],128,"Name","conv2_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv2_2.Bias,"Weights",params.conv2_2.Weights)
    reluLayer("Name","relu2_2")
    maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv3_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_1.Bias,"Weights",params.conv3_1.Weights)
    reluLayer("Name","relu3_1")
    convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_2.Bias,"Weights",params.conv3_2.Weights)
    reluLayer("Name","relu3_2")
    convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv3_3.Bias,"Weights",params.conv3_3.Weights)
    reluLayer("Name","relu3_3")
    maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_1.Bias,"Weights",params.conv4_1.Weights)
    reluLayer("Name","relu4_1")
    convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_2.Bias,"Weights",params.conv4_2.Weights)
    reluLayer("Name","relu4_2")
    convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv4_3.Bias,"Weights",params.conv4_3.Weights)
    reluLayer("Name","relu4_3")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_1.Bias,"Weights",params.conv5_1.Weights)
    reluLayer("Name","relu5_1")
    convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_2.Bias,"Weights",params.conv5_2.Weights)
    reluLayer("Name","relu5_2")
    convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],"WeightL2Factor",0,"Bias",params.conv5_3.Bias,"Weights",params.conv5_3.Weights)
    reluLayer("Name","relu5_3")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    dropoutLayer(0.5,"Name","drop_1")
    fullyConnectedLayer(512,"Name","FC_1")
    eluLayer("Name","ELU_1")
    dropoutLayer(0.5,"Name","drop_2")
    fullyConnectedLayer(512,"Name","FC_2")
    eluLayer("Name","ELU_2")
    fullyConnectedLayer(1,"Name","FC_3")
    regressionLayer('Name','routput')];


%% Define training options


opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 20, ...
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

fprintf(append("\nstart: ", string(datetime),"\n"))

[net, info] = trainNetwork(dataSetTrain, layers, opts);

save("NetworkCNN.mat","info","net")

%% Test

prediction = predict(net, dataSetTest);

%% Calculate accuracy

count = 0;
for i = 1 : length(testY)
    minDiff = testY(i)*0.1;
    diff = abs(testY(i)-prediction(i));
    if(diff<minDiff)
        count = count+1;
    end
end

percentage = (count/length(testY)) * 100;
disp(append("Acuracy: ",string(percentage)))

%% Visualize

for i = 1 : length(imdsTest.Files)
    disp(i)
    img = imread(imgFilesTest(i));
    img = insertShape(img,"circle",[testYfull(i,1), testYfull(i,2), prediction(i)],LineWidth=2);
    
    fig = figure(1);
    imshow(img)

    disp("PAUSED - click enter")
    pause(0.75)
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