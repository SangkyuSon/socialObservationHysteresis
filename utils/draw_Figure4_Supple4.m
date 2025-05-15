function draw_Figure4_Supple4(dataDir)

expNo = 1;
if expNo==1, NoS = 40; else NoS = 14; end
condCols = {[0.8941 0.1020 0.1098],[0.2157 0.4941 0.7216]};


for m = 1:NoS,    
    
    load(fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,m)));

    phaseOfInterest = [1,3]; % inference and pursuit phase
    alignto = 200;
    pixelPerDeg = 33.7409;
    saccadeCrit = 30;
    
    for cnt = 1:length(phaseOfInterest),
        pcell = cellfun(@(x1) x1==phaseOfInterest(cnt),data.phase,'un',0);
        dv{1} = dist2eye(data,'opponent',pcell,alignto);
        dv{2} = dist2eye(data,'avatar',pcell,alignto);
        dv{3} = dist2eye(data,'prey',pcell,alignto);
        ddv = dv{1}-dv{2};

        eyepos = cellfun(@(x1,x2) x1(x2==phaseOfInterest(cnt),:)./pixelPerDeg,data.eyePos,data.phase,'un',0);
        eyespd = cellfun(@(x1) [0;sqrt(sum(diff(x1,[],1).^2,2))*60],eyepos,'un',0);
        eyespd = alignTimeseries(eyespd,alignto,1);

        % collect gaze comparison data
        for b = 1:2,
            blockSel = find(data.block(1:end-1)==b)+1;
            blockSel(blockSel>length(data.block)) = [];

            gazecomp{b,cnt}(m,:) = mean(ddv(blockSel,:),'omitnan')/pixelPerDeg;

            saccadePortion{b,cnt}(m,:) = mean(eyespd(blockSel,:) > saccadeCrit,1,'omitnan');

            blockSel = intersect(blockSel,find(abs( data.friendlinessP - 0.5 ) < 0.1)); % to limit into certain F range
            gazecompCont{b,cnt}(m,:) = mean(ddv(blockSel,:),'omitnan')/pixelPerDeg;

        end
    end

    % correlation with bias
    F = data.friendliness;
    choice = (data.choiceItem+1)./2;
    biasInd = choice-F;
    corrVal(m,:) = corr(ddv(1:end-1,:),biasInd(2:end)','type','Spearman','rows','pairwise');

    % correlation with duration
    dur{m} = sum(ddv(1:end-1,:) < 0,2)/60;
    bias{m} = biasInd(2:end);


    % self-avatar control
    epos = cellfun(@(x1,x2) sqrt(sum((x1-x2).^2,2))./pixelPerDeg,data.eyePos,data.avatar,'un',0);
    ppos = cellfun(@(x1,x2) sqrt(sum((x1-x2).^2,2))./pixelPerDeg,data.eyePos,data.prey,'un',0);
    opos = cellfun(@(x1,x2) sqrt(sum((x1-x2).^2,2))./pixelPerDeg,data.eyePos,data.opponent,'un',0);
    avat = cellfun(@(x1) sum(diff(diff(diff(x1./pixelPerDeg,[],1),[],1),[],1).^2,2,'omitnan'),data.avatar,'un',0);

    pv{m} = cellfun(@(x1) mean(x1(1:end),'omitnan'),ppos);
    ev{m} = cellfun(@(x1) mean(x1(1:end),'omitnan'),opos);
    ov{m} = cellfun(@(x1) mean(x1(1:end),'omitnan'),epos);
    av{m} = cellfun(@(x1) mean(x1(1:end),'omitnan'),avat);

end
    
% Figure 4A
figure
hold on; 
cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'Color',x2),gazecomp(:,2)',condCols)
ylim([-1,1]*7)
ylabel(sprintf('Gaze comparison (deg)'))
xticks([1,60,120,180])
xticklabels({'0','1','2','3'})
xlabel('Time during pursuit (s)')
sgtitle('Figure 4B')

% Figure 4C
figure
shadedplot_custom(corrVal,NoS)
ylabel('Correlation (rho)')
xticks([1,60,120,180])
xticklabels({'0','1','2','3'})
xlabel('Time during pursuit (s)')
sgtitle('Figure 4C')

% Figure 4D
figure
subplot(1,3,1:2)
subjIdx = 12;
corrplot_custom(dur{subjIdx},bias{subjIdx},'verbose',0)
xlabel(sprintf('Duration of\ngaze at opponent (s)'))
ylabel('Bias at next trial (F)')

subplot(1,3,3)
coval = cellfun(@(x1,x2) corr(x1,x2','type','Spearman','rows','complete'),dur,bias)';
bar_custom(coval)
ylim([-1,1]*0.1)
set(gca,'XTick',[])
ylabel('Correlation (rho)')
sgtitle('Figure 4D')

% Figure 4E
figure
subplot(1,3,1:2)
subjIdx = 3;
corrplot_custom(ev{subjIdx},av{subjIdx},'verbose',0)
xlim([0,15])
xlabel(sprintf('Gaze distance \n to player avatar (deg)'))
ylabel(sprintf('Self control unstability (|jerk|)'))

subplot(1,3,3)
coval = cellfun(@(x1,x2,x3,x4) partialcorr(x1',x2',[x3',x4'],'type','Spearman','rows','complete'),ev,av,ov,pv,'un',1)';
bar_custom(coval)
ylim([0,0.6])
set(gca,'XTick',[])
ylabel('Correlation (rho)')
sgtitle('Figure 4E')

% Figure 4F
figure
hold on; 
cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'Color',x2),gazecomp(:,1)',condCols)
ylim([-1,1]*7)
ylabel(sprintf('Gaze comparison (deg)'))
xticks([1,60,120,180])
xticklabels({'0','1','2','3'})
xlabel('Time during inference phase (s)')
sgtitle('Figure 4F')

% Supplementary Figure 4A
figure
hold on; 
cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'Color',x2),gazecompCont(:,2)',condCols)
ylim([-1,1]*7)
ylabel(sprintf('Gaze comparison (deg)'))
xticks([1,60,120,180])
xticklabels({'0','1','2','3'})
xlabel('Time during pursuit (s)')
sgtitle('Supple Figure 4A')

% Supplementary Figure 4B
figure
for p = 1:2,
    subplot(1,2,p)
    hold on;
    cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'Color',x2),saccadePortion(:,p)',condCols)
    ylim([0,0.5])
    ylabel('Saccade portion')
    xticks([1,60,120,180])
    xticklabels({'0','1','2','3'})
    xlabel('Time from event onset (s)')
end
sgtitle('Supple Figure 4B')

end