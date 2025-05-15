function model = taskOptRNN_step1(dataDir)

iterno = 100;

[X, Y] = loadXY(dataDir);

for i = 1:iterno
    fprintf('Iteration (%d/%d)', i, iterno)
    bnet{i} = trainInitialRNN(X, Y);
    Yhat = predictNetwork(bnet{i}, X);
    err(i) = mean((Y - Yhat).^2);
    fprintf('...done\n')
end

[~, bestIdx] = min(err);

model = bnet{bestIdx};

end

function trainedNet = trainInitialRNN(X, Y)

% Setup parameters
inputSize = size(X{1}, 1);  % Number of features
outputSize = 1;
numHiddenUnits = 64;
maxEpochs = 20;  % Adjust as needed
learningRate = 0.01;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
epsilon = 1e-8;

% Initialize average gradient and average squared gradient states
avgGradients = [];
avgSquaredGradients = [];

% Initialize weights
negWeight = 1;
posWeight = 1;

% Define the RNN layers
layers = [ ...
    sequenceInputLayer(inputSize, 'Name', 'input')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'recurrentLayer')
    fullyConnectedLayer(outputSize, 'Name', 'fc')];

% Create dlnetwork object
net = dlnetwork(layers);

% Move network to GPU if available
net = dlupdate(@gpuArray, net);

% Randomly select 300 samples

% Custom training loop
for epoch = 1:maxEpochs

    numSamples = 100;
    indices = randperm(numel(X), numSamples);
    Xep = X(indices);
    Yep = Y(indices);

    % Loop over the training data
    for i = 1:numel(Xep)
        % Convert input to dlarray with appropriate dimensions and move to GPU
        dlX = dlarray(Xep{i}, 'CT'); % 'CT' format: C (Channel), T (Time)
        dlY = dlarray(Yep(i), 'CB');  % 'CB' format: C (Channel), B (Batch)

        % Move to GPU if available
        dlX = gpuArray(dlX);
        dlY = gpuArray(dlY);

        % Evaluate the model gradients and loss using a custom function
        [gradients, loss, negError, posError] = dlfeval(@modelGradients, net, dlX, dlY, negWeight, posWeight);

        % Update the network parameters using the gradients
        [net, avgGradients, avgSquaredGradients] = adamupdate(...
            net, gradients, avgGradients, avgSquaredGradients, i, ...
            learningRate, gradientDecayFactor, squaredGradientDecayFactor, epsilon);
    end

end

% Return the trained network
trainedNet = net;
end

function [gradients, loss, negError, posError] = modelGradients(net, dlX, dlY, negWeight, posWeight)
% Forward pass
dlYPred = forward(net, dlX);

% Compute custom loss that incorporates posError and negError
[loss, posError, negError] = customLoss(dlYPred, dlY, negWeight, posWeight);

% Compute gradients
gradients = dlgradient(loss, net.Learnables);
end

function [loss, posError, negError] = customLoss(YPred, YTrue, negWeight, posWeight)
% Calculate the error
error = YTrue - YPred;

% Calculate positive and negative errors
negError = mean(error(error < 0), 'all');
posError = mean(error(error >= 0), 'all');
negError(isnan(negError)) = 0;
posError(isnan(posError)) = 0;

% Apply different penalties based on the sign of the error
loss = mean((error < 0) .* negWeight .* (error).^2 + ...
    (error >= 0) .* posWeight .* (error).^2, 'all');
end

function predictions = predictNetwork(net, X)
% Convert input to dlarray with appropriate dimensions
dlX = dlarray(X{1}, 'CT'); % 'CT' format: C (Channel), T (Time)

% Move to GPU if available
dlX = gpuArray(dlX);

% Perform prediction
dlYPred = forward(net, dlX);

% Convert dlarray to normal array
predictions = gather(extractdata(dlYPred));
end
