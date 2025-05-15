function draw_Figure3AB_Supple1(dataDir)

expNo = 1;
if expNo==1, NoS = 40; else NoS = 14; end
condCols = {[0.8941 0.1020 0.1098],[0.2157 0.4941 0.7216]};

% collect data
figure
cnt = 0;
maxn = 18; 
binno = 10;
for m = 1:NoS,    
    cnt = cnt + 1;
    load(fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,m)));

    F = data.friendliness(2:end); 
    choice = (data.choiceItem(2:end)+1)/2;
    decisionTime = cellfun(@(x1) sum(x1==2)/60,data.phase); %(s)

    blockIdx = data.block;
    blockFirstTrialIdx = [1,find(diff(data.block)~=0)+1];
    blockLastTrialIdx = [find(diff(data.block)~=0)]+maxn/2;
    for b = 1:2,
        sel = find(blockIdx(1:end-1)==b);
        
        % for Supple Figure 1A
        subplot(5,8,cnt)
        fitout = plotPsychMetric( F(sel), choice(sel), condCols{b} , 1, '-'); hold on;
        
        % for Figure 2A
        groupPsyMet{b}(m,:) = fitout.fity;
        groupbin{b}(m,:) = fitout.biny;
        biasF(m,b) = mean(choice(sel)-F(sel));
        mubin{b}(m,1) = mean(choice(sel));

        % for Supple Figure 1F
        muDeciTime(m,b) = mean(decisionTime(sel));
        [histDeciTime{b}(m,:)] = histvec(decisionTime(sel),F(sel),linspace(0,1,binno+1));


        % for Supple Figure 1D
        for n = 1:maxn, % sum over n+1 trial
            nsel = blockFirstTrialIdx + n-1;
            ncsel = intersect(sel, nsel);
            if sum(ncsel), biasFoverTrial{b}(m,n) = mean(choice(ncsel)-F(ncsel));
            else,          biasFoverTrial{b}(m,n) = nan; end
        end

        % for Supple Figure 1E
        for n = 1:maxn,
            nsel = blockLastTrialIdx + n-1;

            if n <= maxn/2,  ncsel = find(data.block(1:end-1)==b);
            else, ncsel = find(data.block(1:end-1)==(3-b)); end

            ncsel = intersect(ncsel, nsel);

            if sum(ncsel), biasFoverTrialBackward{b}(m,n) = mean(choice(ncsel)-F(ncsel));
            else,          biasFoverTrialBackward{b}(m,n) = nan; end

        end

    end

    sexInfo{m} = data.sex;
    ageInfo(m) = data.age;
end
sgtitle('Supple Figure 1A')

% Figure 3A
figure
subplot(1,3,1:2)
hold on; 
cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'xAxis',fitout.fitx,'Color',x2),groupPsyMet,condCols)
cellfun(@(x1,x2) bar_custom(x1,'LineColor',x2,'errorbar',1,'x',fitout.binx),groupbin,condCols)
xlim([0,1])
xlabel(sprintf('Given F\n(inference phase)'))
xticks(linspace(0,1,3))
ylim([0,1])
ylabel(sprintf('Perceived F\n(=P(choice boost))'))

subplot(1,3,3)
hold on;
bar_custom(biasF(:,1),'scatter',nan,'LineColor',condCols{1},'x',1); % change nan value to see individual participants.
bar_custom(biasF(:,2),'scatter',nan,'LineColor',condCols{2},'x',2);
ylim([-1,1]*0.1)
xticks([1,2])
xticklabels({'Betrayal','Unexp. help'})
ylabel(sprintf('Biases in F\n(perceived F - given F)'))
sgtitle('Figure 3A')

% Figure 3B
figure
entropy = cellfun(@(x1) -(x1.*log2(x1)+(1-x1).*log2(1-x1)),groupbin,'un',0);
muentropy = cellfun(@(x1) mean(x1,2,'omitnan'),entropy,'un',0);

subplot(1,3,1)
hold on;
cellfun(@(x1,x2,x3) bar_custom(x1,'scatter',nan,'LineColor',x2,'x',x3,'errorbar',1), muentropy,condCols,{1,2})
xlim([0.5,2.5])
ylim([0.6,1])
xticks([1.5])
xticklabels('Avg')
ylabel('Entropy (bit)')

subplot(1,3,2:3)
hold on;
cellfun(@(x1,x2) shadedplot_custom(x1,[],'Color',x2),entropy,condCols);
xticks([1,5.5,10])
xticklabels({0,0.5,1})
xlabel('Given F')
ylim([0.6,1])
sgtitle('Figure 3B')

% Supplementary Figure 1B
figure
hold on;
bar_custom(biasF(strcmp(sexInfo,'male'),1),'scatter',nan,'LineColor',condCols{1},'x',1); 
bar_custom(biasF(strcmp(sexInfo,'female'),1),'scatter',nan,'LineColor',condCols{1},'x',2); 
bar_custom(biasF(strcmp(sexInfo,'male'),2),'scatter',nan,'LineColor',condCols{2},'x',3);
bar_custom(biasF(strcmp(sexInfo,'female'),2),'scatter',nan,'LineColor',condCols{2},'x',4);

ylim([-1,1]*0.1)
xticks(1:4)
xticklabels({'M','F','M','F'})
xlabel('Sex')
ylabel(sprintf('Biases in F\n(perceived F - given F)'))
sgtitle('Supple Figure 1B')

% Supplementary Figure 1C
figure;
hold on;
corrplot_custom(ageInfo,biasF(:,1),'Color',condCols{1},'verbose',0)
corrplot_custom(ageInfo,biasF(:,2),'Color',condCols{2},'verbose',0)
xlim([19,28])
xlabel('Age (yr)')
ylim([-0.4,0.2])
ylabel(sprintf('Biases in F\n(perceived F - given F)'))
sgtitle('Supple Figure 1C')

% Supplementary Figure 1D
figure
hold on; 
cellfun(@(x1,x2) shadedplot_custom(x1,[],'Color',x2),biasFoverTrial,condCols);
ylim([-0.2,0.1])
xlabel('Number of repeated trials')
ylabel(sprintf('Biases in F\n(perceived F - given F)'))
sgtitle('Supple Figure 1D')

% Supplementary Figure 1E
figure
hold on; 
twin = [-4:2]+ maxn/2;
cellfun(@(x1,x2) shadedplot_custom(x1(:,twin),[],'Color',x2),biasFoverTrialBackward,condCols);
ylim([-0.2,0.1])
xlabel('N-trials from block change')
ylabel(sprintf('Biases in F\n(perceived F - given F)'))
xticklabels({-5:-1,'N','N+1'})
sgtitle('Supple Figure 1E')

% Supplementary Figure 1F
figure;
subplot(1,3,1)
hold on;
for i = 1:2,
    bar_custom(muDeciTime(:,i),'errorbar',1,'LineColor',condCols{i},'x',i);
end
xlim([0.5,2.5])
ylim([0.8,2])
ylabel('Decision time (s)')
xticks([1.5])
xticklabels('Avg')

subplot(1,3,2:3)
hold on;
cellfun(@(x1,x2) shadedplot_custom(x1,[],'Color',x2),histDeciTime,condCols);
ylim([0.8,2])
xticks([1,5.5,10])
xticklabels({0,0.5,1})
xlabel('Given F')

sgtitle('Supple Figure 1F')

end