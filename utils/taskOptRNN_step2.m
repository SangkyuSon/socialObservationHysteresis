function [data,rawNet] = taskOptRNN_step2(baselineNet,try2use)

if nargin < 2, try2use = 1; end

% Load
[X, Y, Xtest, B] = loadXY(dataDir);

% control variables
numSample = 200;
maxEpochs = 300;
numFolds = 2;

% loop of repetition
rep = 0;
while rep < 1500+1,

    tic;
    data = [];
    rep = rep + 1;
    fprintf('Repetition (%d)\n', rep);

    [XSampled, YSampled, BSampled, XoSampled] = sampleData(X, Y, Xtest, B, numSample);
    
    cvIndices = getCVIdx(numFolds,numSample,BSampled);
    for fold = 1:numFolds,

        fprintf('       Fold (%d/%d)',fold,numFolds)
        
        % Separate training and testing data based on fold
        trainIdx = cvIndices ~= fold;
        testIdx = cvIndices == fold;

        XTrain = XSampled(trainIdx);
        YTrain = YSampled(trainIdx);
        XTest = XSampled(testIdx);
        YTest = YSampled(testIdx);

        % Training on the training set
        [~, ~, loss, lossCV, lastEpoch(fold)] = performTrials(baselineNet, XTrain, XTest, YTrain, YTest, maxEpochs, try2use);

        data.lossCV{fold} = lossCV;
        data.loss{fold} = loss;

        fprintf(' E%d, %.1fm\n', lastEpoch(fold), toc / 60);
    end
    muEpoch = round(mean(lastEpoch));

    [trainedNet, negWeight, loss, lossCV] = performTrials(baselineNet, XSampled, XSampled, YSampled, YSampled, maxEpochs, try2use);

    data.Yhat = predictFromDLNetwork(trainedNet, XoSampled, try2use);
    data.Y = YSampled;
    data.block = BSampled;
    data.negWeight = negWeight';
    data.net = trainedNet;
    
    data(rep).betaBet_step2 = negWeight';
    data(rep).lossCV = lossCV;
    data(rep).loss = loss;
    data(rep).inputF = YSampled;
    data(rep).outputF = predictFromDLNetwork(trainedNet, XoSampled, try2use);
    data(rep).Wih_step2 = trainedNet.net.Layers(2).InputWeights;
    if rep == 1,
        data(rep).Wih_step1 = baselineNet.Layers(2).InputWeights;
    end
    rawNet{rep} = net;
    
end

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

function [trainedNet, negWeightAll, lossAll, lossCV, lastEpoch] = performTrials(baselineNet, XTrain, XTest, YTrain, YTest, maxEpochs, try2use)
    lgraph = layerGraph(baselineNet.Layers);
    lgraph = freezeLayersExcept(lgraph, 'recurrentLayer');
    currentNet = dlnetwork(lgraph);
    
    learningRate = 0.01;
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.999;
    epsilon = 1e-8;
    avgGradients = [];
    avgSquaredGradients = [];
    
    negWeight = dlarray(0.5, 'CB');  % Initialize negWeight
    
    lossAll = nan(maxEpochs, 1);
    lossCV = nan(maxEpochs, 1);
    negWeightAll = nan(maxEpochs, 1);
    
    cnt = 0;
    for epoch = 1:maxEpochs
        totalGradients = [];
        totalLoss = 0;
    
        for i = 1:numel(XTrain)
            dlX = convertToDevice(dlarray(XTrain{i}, 'CT'), try2use);
            dlY = convertToDevice(dlarray(YTrain(i), 'CB'), try2use);
    
            [gradients, loss, negWeightGrad] = dlfeval(@modelGradients, currentNet, dlX, dlY, negWeight);
    
            if isempty(totalGradients)
                totalGradients = gradients;
            else
                for j = 1:height(gradients)
                    totalGradients.Value{j} = totalGradients.Value{j} + gradients.Value{j};
                end
            end
    
            totalLoss = totalLoss + loss;
        end
    
        totalGradients.Value = cellfun(@(x) x / numel(XTrain), totalGradients.Value, 'UniformOutput', false);
        totalLoss = gather(extractdata(totalLoss / numel(XTrain)));
    
        [currentNet, avgGradients, avgSquaredGradients] = adamupdate(...
            currentNet, totalGradients, avgGradients, avgSquaredGradients, epoch, learningRate, gradientDecayFactor, squaredGradientDecayFactor, epsilon);
    
        negWeight = max(min(adamupdate(negWeight, negWeightGrad, [], [], epoch, learningRate, gradientDecayFactor, squaredGradientDecayFactor, epsilon), 1), 0);
    
        YhatTest = predictFromDLNetwork(currentNet, XTest, try2use);
        newLossCV = customLoss(YhatTest,YTest,gather(extractdata(negWeight)));
    
        if (gather(extractdata(negWeight)) >= 0.90 || gather(extractdata(negWeight)) <= 0.10) && (totalLoss < 0.05)
            break;
        end

        lossAll(epoch) = totalLoss;
        lossCV(epoch) = newLossCV;
        negWeightAll(epoch) = gather(extractdata(negWeight));

    end
    
    trainedNet = currentNet;
    lastEpoch = epoch;
    
end

function [gradients, loss, negWeightGrad] = modelGradients(net, dlX, dlY, negWeight)
dlYPred = forward(net, dlX);
loss = customLoss(dlYPred, dlY, negWeight);
gradients = dlgradient(loss, net.Learnables);
negWeightGrad = dlgradient(loss, negWeight);
end

function loss = customLoss(YPred, YTrue, negWeight)
error = YTrue - YPred;
posWeight = 1 - negWeight;
loss = mean((error < 0) .* negWeight .* (error).^2 + (error >= 0) .* posWeight .* (error).^2, 'all');
end

function lgraph = freezeLayersExcept(lgraph, layerNames)
if ischar(layerNames)
    layerNames = {layerNames};
end

for i = 1:numel(lgraph.Layers)
    layer = lgraph.Layers(i);

    if ismember(layer.Name, layerNames)
        if strcmp(layer.Name, 'recurrentLayer')
            if isprop(layer, 'InputWeightsLearnRateFactor')
                layer.InputWeightsLearnRateFactor = 1;
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

function predictedValues = predictFromDLNetwork(currentNet, XInput, try2use)
predictedValues = zeros(1, numel(XInput));
for i = 1:numel(XInput)
    dlX = convertToDevice(dlarray(XInput{i}, 'CT'),try2use);
    dlYPred = forward(currentNet, dlX);
    predictedValues(i) = gather(extractdata(dlYPred));
end
end

function out = convertToDevice(data,try2use)

if try2use & canUseGPU()
    out = gpuArray(data);
else
    out = data;
end

end


function cvIndices = getCVIdx(numFolds,numSample,BSampled)

cvIndices = zeros(1,numSample);
for i = 1:2,
    tmp = repelem(1:numFolds,numSample/2/2);
    cvIndices(find(BSampled==i)) = tmp(randperm(length(tmp)));
end

end