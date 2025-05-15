%% Information
% title              : Observed Betrayal Leaves Hysteresis in Social Inference
% authors            : Sangkyu Son and Seng Bum Michael Yoo
% inquiry about code : Sangkyu Son (ss.sangkyu.son@gmail.com)

%% Set up
clear all; close all; clc; warning off;
genDir = pwd;                            % Note, change this line into proper working directory
utilDir = fullfile(genDir,'/utils');
dataDir = fullfile(genDir,'/data');
addpath(genpath(utilDir))

%% Figures
draw_Figure2(dataDir);            % Exp1, basic analysis
draw_Figure3AB_Supple1(dataDir);  % Exp1, Asymetrical bias related 
draw_Figure3EG_Supple23(dataDir); % Exp1, Engergy landscape related
draw_Figure4_Supple4(dataDir);    % Exp1, Gaze shift related
draw_Figure5_Supple5(dataDir);    % Exp1, task-optimzed RNN
draw_Figure6_Supple6(dataDir);    % Exp2, hysteresis related