function data = loadTaskOptRNN(dataDir)

dataFile = fullfile(dataDir,'data_taskOptRNN.mat');
try2use = 0; % change into 1 this if GPU is possible
if ~exist(dataFile),
    step1Model = taskOptRNN_step1(dataDir);
    [data,rawNet] = taskOptRNN_step2(step1Model,try2use);
    data = taskOptRNN_step3(dataDir, data, step1Model, rawNet, try2use);
    save(dataFile,'data')
else
    load(dataFile);
end

end
