function data = autoLogReg_exp2(dataDir,subNo)

expNo = 2;
fileName = fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,subNo));
load(fileName);

if ~isfield(data,'Fhat')

    phaseIdx = 1;

    nT = 360;
    winSz = 60;
    winNo = 5;

    avat = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.avatar,data.phase,'un',0),nT,-1);
    oppo = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.opponent,data.phase,'un',0),nT,-1);
    prey = alignTimeseries(cellfun(@(x1,x2) x1(x2==phaseIdx,:),data.prey,data.phase,'un',0),nT,-1);

    a2p = sqrt(sum((prey-avat).^2,3));
    o2p = sqrt(sum((prey-oppo).^2,3));
    o2a = sqrt(sum((avat-oppo).^2,3));

    X = cat(3,a2p,o2p,o2a); xn = size(X,3);
    Y = boolean((data.choiceItem'+1)/2);
    F = [0,data.fchg(1:end-1)]';
    B = data.block;

    for b = 1:2,
        bsel = B == b;
        n = sum(bsel);

        rX = [];
        for t = winSz:winSz:nT
            rX = cat(1,rX,reshape(movmeanNresamp(X(bsel,(-winSz+1:0)+t,:),winSz/winNo,2),n,[]));
        end
        rn = size(rX,1)/n;

        rF = repmat(F(bsel),rn,1);
        rY = repmat(Y(bsel),rn,1);

        coef{b} = glmfit(cat(2,rX,rF),rY,'binomial','link','logit');
    end

    for k = 1:length(data.block),
        try
            kcoef = coef{data.block(k)};
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
                tX = transferVal(tX,nan,0);

                tY{k}(t) = glmval(kcoef,cat(2,tX,F(k)),'logit');
            end
        end
    end

    data.Fhat = tY;
    save(fileName,'data');
end

end