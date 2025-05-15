function draw_Figure2(dataDir)

expNo = 1;
if expNo==1, NoS = 40; else NoS = 14; end
condCols = {[0.8941 0.1020 0.1098],[0.2157 0.4941 0.7216]};

% collect data
for m = 1:NoS,
    
    load(fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,m)));
    
    binNo = 3;
    binEdge = linspace(0,1,binNo+1);

    reward = data.earnedMoney*2;
    choice = (data.choiceItem+3)/2;

    for c = 1:2,
        decisionTimeSummary{c}(m,:) = histvec(reward(choice==c),data.friendliness(choice==c),binEdge);

        for b = 1:2,
            mfr(m,c+(b-1)*2) = mean(reward(choice==c & ((sign(data.fchg)+3)/2)==b));
        end
    end
end

% Figure 2F
figure
hold on; cellfun(@(x1,x2) bar_custom(x1,'errorbar',1,'LineColor',[0,0,0],'LineStyle',x2),decisionTimeSummary,{'-','--'});
xticks([1,2,3])
xticklabels({0,0.5,1})
xlabel('Given F')
ylim([80,180])
ylabel('Reward (won/trial)')
sgtitle('Figure 2F')

% Figure 2G
figure
moneyLoss = mfr(:,2)-mfr(:,1);
moneyGain = mfr(:,4)-mfr(:,3);

hold on;
bar_custom(moneyLoss,'scatter',nan,'LineColor',condCols{1},'x',1); % change nan value to see individual participants.
bar_custom(moneyGain,'scatter',nan,'LineColor',condCols{2},'x',2);
ylim([-1,1]*120)
xticks([1,2])
xticklabels({'Betrayal','Unexp. help'})
ylabel('Reward loss/gain (won/trial)')
sgtitle('Figure 2G')

end