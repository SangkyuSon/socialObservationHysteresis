function draw_Figure3EG_Supple23(dataDir)

expNo = 1;
if expNo==1, NoS = 40; else NoS = 14; end
condCols = {[0.8941 0.1020 0.1098],[0.2157 0.4941 0.7216]};

% collect data
xn = 20;
xEdge = linspace(0,1,xn+1);
x = movmean(xEdge,2);
x = x(2:end);
findx = linspace(0,1,1e5);
FbinNo = 3;

for m = 1:NoS,
    data = autoLogReg_exp1(dataDir,m);

    B = [0,data.block(1:end-1)];
    Fhat = cellfun(@(x1) x1(31:end-30),data.Fhat,'un',0);

    dY = cellfun(@(x1) (diff(x1)),Fhat,'un',0);
    Fhat = cellfun(@(x1) x1(1:end-1),Fhat,'un',0);
    F = data.friendliness;
    mdY = cellfun(@(x1,x2) histvec(x1,x2,xEdge),dY,Fhat,'un',0);

    ebin = linspace(0,1,FbinNo+1);

    for e = 1:FbinNo,
        esel = (F >= ebin(e)) & (F <= ebin(e+1));
        for b = 1:2,
            trsel = esel & (B==b);
            edY = mean(cell2mat(mdY(trsel)),2,'omitnan');
            tmpF = cumtrapz(transferVal(edY,nan,0));

            basinDepth{b}(m,e) = range(tmpF); % basin depth
            
            tmpF = normalize(tmpF,'range');
            energy{e,b}(m,:) = tmpF;

            polyout = polyval(fitpoly(x,tmpF,[2,4]),findx);
            [~,minIdx] = min(polyout);
            fxpnt{e}(m,b) = findx(minIdx);

            basinSteep{b}(m,e) = max( [...
                estslope(fliplr(polyout(1:minIdx)))*minIdx, ...
                estslope(polyout(minIdx:end)*(length(polyout)-minIdx)) ] );
        end
    end

    for b = 1:2,
        simul{b}(m,:,:,:) = data.simulPsych(:,:,:,b);
        psych{b}(m,:,:) = data.psych{b}; 
    end

end

% Figure 3E
figure;
for t = 2:4,
    subplot(1,3,t-1)
    hold on; 
    cellfun(@(x1,x2) shadedplot_custom(x1(:,:,t),NoS,'xAxis',linspace(0,1,100),'Color',x2),psych,condCols)
    ylim([0,1])
    xlabel('Given F')
    ylabel('Inferred F')
    title(sprintf('-%.0fs',4-t))
end
sgtitle(sprintf('Figure 3E\n(time from item decision period)'))


% Figure 3G
figure;
Fnames = {'Low','Medium','High'};
for e = 1:FbinNo,
    subplot(2,FbinNo,e)
    hold on; cellfun(@(x1,x2) shadedplot_custom(x1,NoS,'xAxis',x,'Color',x2),energy(e,:),condCols)
    xticks([0,0.5,1])
    xlim([0,1])
    xlabel('Inferred $\hat{f}$','Interpreter','latex')
    ylim([0,1])
    ylabel('Norm. energy')
    subtitle(sprintf('Given F:\n%s',Fnames{e}))

    subplot(2,FbinNo,e+FbinNo)
    hold on;
    bar_custom(fxpnt{e}(:,1),'scatter',nan,'LineColor',condCols{1},'x',1); 
    bar_custom(fxpnt{e}(:,2),'scatter',nan,'LineColor',condCols{2},'x',2);
    
    ylim([0,1])
    xlim([0.5,2.5])
    xticks([1,2])
    xticklabels({'Betrayal','Unexp. help'})
end
sgtitle('Figure 3G')

% Supplementary Figure 2
figure;
[~,~,e1n,e2n] = size(simul{1});
for e1 = 1:e1n
    for e2 = 1:e2n

        subplot(e2n,e1n,e2+(e1-1)*e2n)
        hold on
        cellfun(@(x1,x2) shadedplot_custom(x1(:,:,e1,e2),[],'Color',x2,'xAxis',linspace(0,1,100)),simul,condCols)
        ylim([0,1])

        xlabel(sprintf('Given F\n%.f%%',(e2-1)*1/3*100));
        ylabel(sprintf('Inferred F\n%.f%%',(e1-1)*100));
    end
end
sgtitle('Supple Figure 2')

% Supplementary Figure 3
figure
subplot(1,2,1)
hold on; cellfun(@(x1,x2) bar_custom(x1,'errorbar',1,'LineColor',x2),basinDepth,condCols)
ylim([0,15]*1e-2)
ylabel('Basin depth (a.u.)')
xticklabels({'L','M','H'})
xlabel('Given F range')

subplot(1,2,2)
hold on; cellfun(@(x1,x2) bar_custom(x1,'errorbar',1,'LineColor',x2),basinSteep,condCols)
ylim([0.9,1.4])
ylabel('Steepness (a.u.)')
xticklabels({'L','M','H'})
xlabel('Given F range')
sgtitle('Supple Figure 3')



end