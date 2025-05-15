function data = taskOptRNN_step3(dataDir, data, baselineNet, rawNet, try2use)

if nargin < 5, try2use = 1; end

% Load
[X, Y, Xtest, B] = loadXY(dataDir);

% control variables
maxEpochs = 35;
numSample = 1000;
outDir = fullfile(dataDir, 'processed', getCatName);
poolDir = fullfile(outDir,'pool');

% loop of transformation
for rep = 1:length(rawNet)

    
    [XSampled, YSampled, BSampled, XoSampled] = sampleData(X, Y, Xtest, B, numSample);
    negWeight = data.betaBet_step2(sum(~isnan(data.betaBet_step2)));
    
    ampFac = [-1,1,-1];
    [negWeightHist,trainedNet] = performTrials(rawNet{rep}, baselineNet, ampFac, XSampled, YSampled, negWeight, maxEpochs, try2use);

    data(rep).betaBet_step3 = [nan(length(negWeight),1);negWeightHist,nan(300-nan(length(negWeight),1)-length(negWeightHist))];
    
end

end

function [negWeightAll,currentNet] = performTrials(biasedNet, baselineNet, amplifier, X, Y, negWeight, maxEpochs, try2use)

lgraph = layerGraph(baselineNet.Layers);

recurrentLayer = lgraph.Layers(strcmp({lgraph.Layers.Name}, 'recurrentLayer'));
replaceInput = baselineNet.Layers(2).InputWeights + (biasedNet.Layers(2).InputWeights - baselineNet.Layers(2).InputWeights) .* repelem(amplifier,2);
recurrentLayer.InputWeights = replaceInput;
lgraph = replaceLayer(lgraph, 'recurrentLayer', recurrentLayer);

lgraph = freezeLayersExcept(lgraph, {'recurrentLayer', 'inputLayer'});  % Freeze both RNN input and recurrent layers
currentNet = dlnetwork(lgraph);

learningRate = 0.05;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
epsilon = 1e-8;
avgGradients = [];
avgSquaredGradients = [];

origNegWeight = negWeight;
negWeight = dlarray(negWeight, 'CB');  % Initialize negWeight

negWeightAll = nan(maxEpochs, 1);
for epoch = 1:maxEpochs

    totalLoss = 0;

    for i = 1:numel(X)
        dlX = convertToDevice(dlarray(X{i}, 'CT'), try2use);
        dlY = convertToDevice(dlarray(Y(i), 'CB'), try2use);

        [loss, negWeightGrad] = dlfeval(@modelGradients, currentNet, dlX, dlY, negWeight);

        totalLoss = totalLoss + loss;
    end

    totalLoss = gather(extractdata(totalLoss / numel(X)));
    negWeight = max(min(adamupdate(negWeight, negWeightGrad, [], [], epoch, learningRate, gradientDecayFactor, squaredGradientDecayFactor, epsilon), 1), 0);
    negWeightAll(epoch) = gather(extractdata(negWeight));

    
    if ( (origNegWeight > 0.5) & (negWeight < 0.1) ) | ( (origNegWeight < 0.5) & (negWeight > 0.9) ),
        break
    end
       
end


end

function [loss, negWeightGrad] = modelGradients(net, dlX, dlY, negWeight)
dlYPred = forward(net, dlX);
loss = customLoss(dlYPred, dlY, negWeight);
negWeightGrad = dlgradient(loss, negWeight);  % Calculate gradient only for negWeight
end

function lgraph = freezeLayersExcept(lgraph, layerNames)
if ischar(layerNames)
    layerNames = {layerNames};
end

for i = 1:numel(lgraph.Layers)
    layer = lgraph.Layers(i);

    if ismember(layer.Name, layerNames)
        if strcmp(layer.Name, 'recurrentLayer') || strcmp(layer.Name, 'inputLayer')
            if isprop(layer, 'InputWeightsLearnRateFactor')
                layer.InputWeightsLearnRateFactor = 0;
            end
            if isprop(layer, 'RecurrentWeightsLearnRateFactor')
                layer.RecurrentWeightsLearnRateFactor = 0;
            end
            if isprop(layer, 'BiasLearnRateFactor')
                layer.BiasLearnRateFactor = 0;
            end
        end
    else
        if isprop(layer, 'WeightLearnRateFactor')
            layer.WeightLearnRateFactor = 0;
        end
        if isprop(layer, 'BiasLearnRateFactor')
            layer.BiasLearnRateFactor = 0;
        end
    end

    lgraph = replaceLayer(lgraph, layer.Name, layer);
end
end

function todoList = getTodo(inputDir, outDir)

% Get list of rep files in inputDir
inputFiles = dir(fullfile(inputDir, 'rep_*.mat'));
inputNumbers = [];

% Extract numbers from input files
for i = 1:length(inputFiles)
    % Extract the number from the file name (e.g., rep_0001.mat -> 1)
    numberStr = erase(erase(inputFiles(i).name, 'rep_'), '.mat');
    inputNumbers = [inputNumbers, str2double(numberStr)];
end

% Get list of trans files in outDir
outputFiles = dir(fullfile(outDir, 'trans_*.mat'));
outputNumbers = [];

% Extract numbers from output files
for i = 1:length(outputFiles)
    % Extract the number from the file name (e.g., trans_0001.mat -> 1)
    numberStr = erase(erase(outputFiles(i).name, 'trans_'), '.mat');
    outputNumbers = [outputNumbers, str2double(numberStr)];
end

% Find numbers present in inputNumbers but not in outputNumbers
todoList = setdiff(inputNumbers, outputNumbers);

end


function [XTrain, YTrain, BTrain, XoTrain] = sampleData(X, Y, Xtest, B, no)
idxB1 = find(B == 1);
idxB2 = find(B == 2);

selectedIdxB1 = datasample(idxB1, no / 2, 'Replace', false);
selectedIdxB2 = datasample(idxB2, no / 2, 'Replace', false);
selectedIdx = [selectedIdxB1; selectedIdxB2];
selectedIdx = selectedIdx(randperm(no));

XTrain = Xtest(selectedIdx);
YTrain = Y(selectedIdx);
BTrain = B(selectedIdx);
XoTrain = X(selectedIdx);
end


function loss = customLoss(YPred, YTrue, negWeight)
error = YTrue - YPred;
posWeight = 1 - negWeight;
loss = mean((error < 0) .* negWeight .* (error).^2 + (error >= 0) .* posWeight .* (error).^2, 'all');
end

function out = convertToDevice(data,try2use)

if try2use & canUseGPU()
    out = gpuArray(data);
else
    out = data;
end

end


function predictedValues = predictFromDLNetwork(currentNet, XInput, try2use)
predictedValues = zeros(1, numel(XInput));
for i = 1:numel(XInput)
    dlX = convertToDevice(dlarray(XInput{i}, 'CT'),try2use);
    dlYPred = forward(currentNet, dlX);
    predictedValues(i) = gather(extractdata(dlYPred));
end
end