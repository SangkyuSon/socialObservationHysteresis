function draw_Figure5_Supple5(dataDir)

condCols = {[0.8941 0.1020 0.1098],[0.2157 0.4941 0.7216]};

data = loadTaskOptRNN(dataDir);

inputF = reshape([data.inputF],length(data(1).inputF),[]);
outputF = reshape([data.outputF],length(data(1).outputF),[]);
biasF = mean(outputF-inputF,1);

betaBet = reshape([data.betaBet_step2],length(data(1).betaBet_step2),[])';
clear nval
for k = 1:size(betaBet,1), 
    nval(k) = betaBet(k,sum(~isnan(betaBet(k,:)))); 
end
betSel = nval > 0.89;
uehSel = nval < 0.11;

% Figure 5B
figure;
plot(betaBet(betSel,:)','Color',condCols{1})
hold on;
plot(betaBet(uehSel,:)','Color',condCols{2})
xlabel('Epoch')
ylabel('Sensitivity to Betrayal')
sgtitle('Figure 5B')

% Figure 5C and D
figure;
subplot(1,2,1)
histogram(reshape(inputF(:,uehSel),[],1),'FaceColor',condCols{1},'Normalization','probability')
hold on;
histogram(reshape(inputF(:,betSel),[],1),'FaceColor',condCols{2},'Normalization','probability')
xlabel('Given F')
ylabel('Probability')
title('Figure 5C')

subplot(1,2,2)
histogram(reshape((biasF(:,uehSel)),[],1),'Normalization','probability','FaceColor',condCols{2})
hold on;
histogram(reshape((biasF(:,betSel)),[],1),'Normalization','probability','FaceColor',condCols{1})
xlabel('Biases in F')
ylabel('Probability')
title('Figure 5D')

% Figure 5E
Wih_step2 = arrayfun(@(x) x.Wih_step2, data, 'UniformOutput', false);
Wih_step2 = permute(cat(3, Wih_step2{:}),[3,1,2]);
Wih_step1 = double(permute(data(1).Wih_step1,[3,1,2]));
delta_Wih = Wih_step2 - Wih_step1;


delta_Wih_aligned = double(delta_Wih);
for c = 1:size(delta_Wih_aligned,2),
    for t = 1:size(delta_Wih_aligned,3),
        delta_Wih_aligned(:,c,t) = delta_Wih(:,c,t) * sign(corr(delta_Wih_aligned(:,c,t),mean(biasF,1)','type','Spearman'));
    end
end

figure
subplot(2,3,1); corrplot_custom(delta_Wih_aligned(:,1,1),biasF,'verbose',0);
subplot(2,3,2); corrplot_custom(delta_Wih_aligned(:,50,3),biasF,'verbose',0);
subplot(2,3,4); corrplot_custom(delta_Wih_aligned(:,3,1),biasF,'verbose',0);
xlabel('$\Delta W\_{ih}$','Interpreter','latex')
ylabel('Bias in F')
subplot(2,3,5); corrplot_custom(delta_Wih_aligned(:,230,3),biasF,'verbose',0);

subplot(3,3,[6,9])
[coval,p] = corr(reshape(delta_Wih_aligned,size(delta_Wih_aligned,1),[]),biasF','type','Spearman');
q = mafdr(p(:));
notsig = ~(p < 0.05 & q < 0.05);
ed = linspace(0,0.4,20);
histogram(coval,ed,'Normalization','probability','FaceColor',[0,0,0])
hold on;
tmp = coval; tmp(~notsig) = nan;
histogram(tmp,ed,'Normalization','probability','FaceColor',[1,1,1]*0.95)
xlabel('Correlation (rho)')
ylabel('Probability')
sgtitle('Figure 5E')


% Figure 5F

delta_Wih_BetNet = (squeeze(mean((delta_Wih_aligned(betSel,:,:)),2)));
delta_Wih_uehNet = (squeeze(mean((delta_Wih_aligned(uehSel,:,:)),2)));
delta_Wih_BetNet = double([mean(delta_Wih_BetNet(:,1:2),2),mean(delta_Wih_BetNet(:,5:6),2)]);
delta_Wih_uehNet = double([mean(delta_Wih_uehNet(:,1:2),2),mean(delta_Wih_uehNet(:,5:6),2)]);

figure
hold on;
xbar_custom(delta_Wih_BetNet(:,1),delta_Wih_BetNet(:,2),'scatterColor',condCols{1},'unityLine',0)
xbar_custom(delta_Wih_uehNet(:,1),delta_Wih_uehNet(:,2),'scatterColor',condCols{2},'unityLine',0)
yline(0,'Color',[1,1,1]*0.6,'LineStyle','--')
xline(0,'Color',[1,1,1]*0.6,'LineStyle','--')
xlim([-1,1]*0.02)
ylim([-1,1]*0.02)
xlabel('$\Delta W_{ih}^{plyaer}$','Interpreter','latex')
ylabel('$\Delta W_{ih}^{opponent}$','Interpreter','latex')
sgtitle('Figure 5F')

% Figure 5H

betaBet_step3 = reshape([data.betaBet_step3],length(data(1).betaBet_step3),[])';

figure;
subplot(2,1,1)
plot(betaBet(betSel,:)','Color',min(condCols{1}*7,1))
hold on;
plot(betaBet_step3(betSel,:)','Color',condCols{1})
xlim([0,250])
xlabel('Epoch')
ylabel('Sensitivity to Betrayal')

subplot(2,1,2)
plot(betaBet(uehSel,:)','Color',min(condCols{2}*1.5,1))
hold on;
plot(betaBet_step3(uehSel,:)','Color',condCols{2})
xlim([0,250])
xlabel('Epoch')
ylabel('Sensitivity to Betrayal')
sgtitle('Figure 5H')

% Supplementary Figure 5
figure;
cvACC = 1-cell2mat(arrayfun(@(x1) nanmean(alignTimeseries(cellfun(@(x2) x2(1:sum(~isnan(x2))),x1.lossCV,'un',0),300,1),1),data,'un',0)');
ACC = 1-cell2mat(arrayfun(@(x1) nanmean(alignTimeseries(cellfun(@(x2) x2(1:sum(~isnan(x2))),x1.loss,'un',0),300,1),1),data,'un',0)');

shadedplot_custom(cvACC,[],'std',5,'Color',[255,165,0]/255)
hold on;
shadedplot_custom(ACC,[],'std',5)
xlabel('Epoch')
ylabel('Accuracy of training / test dataset')
sgtitle('Supple Figure 5')


end


