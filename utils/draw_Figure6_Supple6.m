function draw_Figure6_Supple6(dataDir)

expNo = 2;
if expNo==1, NoS = 40; else NoS = 14; end
eno = 7;
condCols = getColor(eno);
pixelPerDeg = 33.7409;

% collect data
for m = 1:NoS,

    data = autoLogReg_exp2(dataDir,m);

    % item choice
    F = data.friendliness;
    C = (data.choiceItem+1)/2;
    fc = [0,data.fchg(1:end-1)];
    B = [0,data.block(1:end-1)];
    
    ed = linspace(-0.45,0.45,eno+1);
    x = movmean(ed,2); x = x(2:end);

    for e = 1:eno,
        for b = 1:2,
            esel = fc >= ed(e) & fc <= ed(e+1) & B==b;
            
            if sum(esel),
                fout = psychMetric(F(esel),C(esel),'MaxFunEvals',1e2);
                psy{e,b}(m,:) = fout.fity;
                oe{b}(m,e) = mean(C(esel)-F(esel));
            end
        end
    end
    
    % gaze location
    alignto = 200;
    pIdx = 3;
    pcell = cellfun(@(x1) x1==pIdx,data.phase,'un',0);
    dv{1} = dist2eye(data,'opponent',pcell,alignto);
    dv{2} = dist2eye(data,'avatar',pcell,alignto);
    dv{3} = dist2eye(data,'prey',pcell,alignto);
    ddv = dv{1} - dv{2};
    
    for e = 1:eno,
        for b = 1:2,
            esel = fc >= ed(e) & fc <= ed(e+1) & data.block==b;
            gazeComp{b,e}(m,:) = nanmean(ddv(esel,:),1)/pixelPerDeg;
        end
    end

    % energy landscape
    xn = 20;
    dtyno = 3;
    gno = 7;
    xx = linspace(0,1,xn+1);
    x = movmean(xx,2);
    x = x(2:end);
    findx = linspace(0,1,1e5);

    gbin = linspace(-0.45,0.45,gno+1);
    gval = movmean(gbin,2); gval = gval(2:end);
    B = data.block;
    F = [0,data.fchg(1:end-1)];

    data2use = cellfun(@(x1) x1(31:end-30),data.Fhat,'un',0);

    dY = cellfun(@(x1) (diff(x1)),data2use,'un',0);
    data2use = cellfun(@(x1) x1(1:end-1),data2use,'un',0);
    mdY = cellfun(@(x1,x2) histvec(x1,x2,xx),dY,data2use,'un',0);

    ebin = linspace(0,1,dtyno+1);
    for e = 1:dtyno,
        esel = (data.friendliness >= ebin(e)) & (data.friendliness <= ebin(e+1));
        for b = 1:2,
            bsel = B==b;
            for g = 1:gno,

                gsel = F >= gbin(g) & F <=gbin(g+1);

                trsel = esel & bsel & gsel;
                edY = mean(cell2mat(mdY(trsel)),2,'omitnan');
                tmpF = normalize(cumtrapz(transferVal(edY,nan,0)),'range');
                eF{e,b,g}(m,:) = tmpF;
                try,
                    [~,minIdx] = min(polyval(fitpoly(x,tmpF,[3,4]),findx));
                    fxpnt{e,g}(m,b) = findx(minIdx);
                end
            end
        end
    end
end
eF = cellfun(@(x1) transferVal(x1,0,nan),eF,'un',0);
 
eno = 7;
% Figure 6D and E
figure;
for b = 1:2,

    if b==1, lty = '-'; else lty = '--'; end
    
    for e = 1:eno,
        subplot(2,4,e)
        shadedplot_custom(psy{e,b},[],'Color',condCols{e},'LineStyle',lty,'xAxis',fout.fitx)
        ylim([0,1])
        xlim([0,1])
        xlabel('Given F')
        ylabel('Perceived F')
        title(sprintf('%.02f',x(e)))
        hold on;
    end

    subplot(2,4,eno+1)
    plot(1:eno,mean(oe{b},1),'Color',[1,1,1]*0);hold on;
    for e = 1:eno,
        bar_custom(oe{b}(:,e),'x',e,'LineColor',condCols{e},'errorbar',1)
    end
    xlim([0.5,7.5])
    ylim([-1,1]*0.15)
    xticks([1,4,7])
    xticklabels({'-0.39','dF=0','0.39'})
    ylabel('Bias in F')
end
sgtitle('Figure 6D and E')

% Figure 6F and G
figure;
mdvs = cellfun(@(x1) mean(x1(:,91:180),2),gazeComp,'un',0);
for b = 1:2,

    if b==1, lty = '-'; else lty = '--'; end
    
    for e = 1:eno,
        subplot(2,4,e)
        shadedplot_custom(gazeComp{b,e},[],'Color',condCols{e},'LineStyle',lty,'xAxis',linspace(0,200/60,200))
        ylim([-1,1]*6)

        xlabel('Time during pursuit (s)')
        ylabel('Gaze comparison (deg)')
        title(sprintf('%.02f',x(e)))
        hold on;
    end

    subplot(2,4,eno+1)
    
    plot(1:eno,cellfun(@mean,mdvs(b,:)),'Color',[1,1,1]*0);hold on;
    for e = 1:eno,
        bar_custom(mdvs{b,e},'x',e,'LineColor',condCols{e},'errorbar',1)
    end
    xlim([0.5,7.5])
    ylim([-1,1]*6)
    xticks([1,4,7])
    xticklabels({'-0.39','dF=0','0.39'})
    ylabel('Gaze comparison (deg)')
end
sgtitle('Figure 6F and G')

% Figure 6H and I
figure
for b = [1,2],
    subplot(1,3,b)
    data = squeeze(mean(squeeze(cell2mat(eF(2,b,:))),1,'omitnan'));
    surf(data)
    colormap(flipud(gray))

    view(80,65)
    axis square
    yticks([1,10.5,20])
    ylim([1,20])
    yticklabels([0,0.5,1])
    ylabel('Inferred F')
    xticks([1,4,7])
    xticklabels({'-0.39','dF=0','0.39'})
    zlabel('Norm. Energy')

    subplot(1,3,3)
    plot(1:eno,cellfun(@(x) mean(x(:,b),1),fxpnt(2,:)),'Color',[1,1,1]*0);hold on;
    for e = 1:eno,
        bar_custom(fxpnt{2,e},'x',e,'LineColor',condCols{e},'errorbar',1)
    end
    xlim([0.5,7.5])
    ylim([0.2,0.8])
    xticks([1,4,7])
    xticklabels({'-0.39','dF=0','0.39'})
    ylabel('Stable fixed point')
    axis square

end
sgtitle('Figure 6H and I')

end