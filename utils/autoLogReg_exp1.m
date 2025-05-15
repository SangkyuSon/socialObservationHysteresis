function data = autoLogReg_exp1(dataDir,subNo)

expNo = 1;
fileName = fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,subNo));
load(fileName);

if ~isfield(data,'Fhat')

    phaseIdx = 1; % inference phase 
    n = length(data.fchg);
    nT = 360;   % maximum 6s
    winSz = 60; % 60 frames
    winNo = 5;  % divide 60 frames into 5 (=each time bin to be 12 frames)

    avat = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.avatar,data.phase,'un',0),nT,-1);
    oppo = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.opponent,data.phase,'un',0),nT,-1);
    prey = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.prey,data.phase,'un',0),nT,-1);

    a2p = sqrt(sum((prey-avat).^2,3));
    o2p = sqrt(sum((prey-oppo).^2,3));
    o2a = sqrt(sum((avat-oppo).^2,3));

    X = cat(3,a2p,o2p,o2a); xn = size(X,3);
    Y = boolean((data.choiceItem'+1)/2);
    F = [0,data.fchg(1:end-1)]';
    
    rX = [];
    for t = winSz:winSz:nT
        rX = cat(1,rX,reshape(movmeanNresamp(X(:,(-winSz+1:0)+t,:),winSz/winNo,2),n,[]));
    end
    rn = size(rX,1)/n;
    rF = repmat(F,rn,1);
    rY = repmat(Y,rn,1);

    coef = glmfit(cat(2,rX,rF),rY,'binomial','link','logit');
    cn = length(coef);
    
    % time series estimation
    for k = 1:n,

        tsel = data.phase{k}==phaseIdx;
        avat = data.avatar{k}(tsel,:);
        oppo = data.opponent{k}(tsel,:);
        prey = data.prey{k}(tsel,:);

        a2p = sqrt(sum((prey-avat).^2,2));
        o2p = sqrt(sum((prey-oppo).^2,2));
        o2a = sqrt(sum((avat-oppo).^2,2));

        X = cat(2,a2p,o2p,o2a);
        xn = size(X,2);

        mX = cat(1,zeros(winSz/2,xn),X,zeros(winSz/2,xn));
        for t = 1:size(X,1),

            tX = reshape(movmeanNresamp(mX(t+(0:winSz-1),:),winSz/winNo,1),1,[]);

            Fhat{k}(t) = glmval(coef,cat(2,tX,F(k)),'logit');

        end
    end
    data.Fhat = Fhat;
    

    % psychometric curve estimation
    nT = 300;
    binsz = 60;
    FhatAlign = alignTimeseries(Fhat,nT,-1);
    FhatAlign(:,1:30) = [];
    FhatAlign(:,end-30+1:end) = [];
    [~,nT] = size(FhatAlign);

    F = [0,data.block(1:end-1)];
    for b = 1:2,

        for t = 1:nT/binsz,
            twin = ((t-1)*binsz+1):(t*binsz);
            tY = nanmean(FhatAlign(F==b,twin),2);
            fout = psychMetric(data.friendliness(F==b),tY');
            psych{b}(:,t) = fout.fity;
        end
    end
    data.psych = psych;


    % parameter simulation
    ed1 = -1:2;
    eno = length(ed1);
    ed2 = linspace(-1,0,eno);
    B = [1,data.block(1:end-1)];

    nT = 60;
    winSz = 60;
    winNo = 5;

    avat = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.avatar,data.phase,'un',0),nT,-1);
    oppo = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.opponent,data.phase,'un',0),nT,-1);
    prey = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.prey,data.phase,'un',0),nT,-1);

    a2p = sqrt(sum((prey-avat).^2,3));
    o2p = sqrt(sum((prey-oppo).^2,3));
    o2a = sqrt(sum((avat-oppo).^2,3));

    X = cat(3,a2p,o2p,o2a); xn = size(X,3);
    rX = reshape(movmeanNresamp(X,winSz/winNo,2),n,[]);
    F = [0,data.fchg(1:end-1)]';

    for e1 = 1:eno
        for e2 = 1:eno,
            ecoef = [coef(1);coef(2:end-1).*(1+ed2(e2));coef(end)*(1+ed1(e1))];
            eY = glmval(ecoef,cat(2,rX,F),'logit');
            for b = 1:2,
                fout = psychMetric(data.friendliness(B==b),eY(B==b)');
                fy(:,e1,e2,b) = fout.fity;
            end
        end
    end
    data.simulPsych = fy;

    save(fileName,'data');
end

end