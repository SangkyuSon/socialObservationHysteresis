function [X, Y, Xtest, B] = loadXY(dataDir)

expNo = 1;
if expNo==1, NoS = 40; else NoS = 14; end

X = []; Y = []; Xtest = []; B = [];
for m = 1:NoS,

    load(fullfile(dataDir,sprintf('data_exp%d_S%02d.mat',expNo,m)));

    cen = cellfun(@(x1,x2,x3) (x1+x2+x3)/3, data.avatar, data.opponent, data.prey, 'un', 0);

    phaseIdx = 1;
    adata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.avatar, cen, 'un', 0);
    odata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.opponent, cen, 'un', 0);
    pdata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.prey, cen, 'un', 0);
    F = data.friendliness;

    x = cellfun(@(x1,x2,x3) cat(1, x1', x2', x3'), adata, pdata, odata, 'un', 0);
    y = F;

    X = cat(2, X, x);
    Y = cat(2, Y, y);

    phaseIdx = 3;
    adata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.avatar, cen, 'un', 0);
    odata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.opponent, cen, 'un', 0);
    pdata = cellfun(@(x1,x2,x3) x2(x1==phaseIdx,:)-x3(x1==phaseIdx,:), data.phase, data.prey, cen, 'un', 0);
    xtest = cellfun(@(x1,x2,x3) cat(1, x1', x2', x3'), adata, pdata, odata, 'un', 0);

    Xtest = cat(2, Xtest, xtest);
    B = cat(2, B, data.block);
end

Y = Y';
B = B';
end
