function [distVal,szVal] = dist2eye(data,comppos,tIdx,alignto)
if nargin < 4, alignto = -100; end
ad = cellfun(@(x1,x2) sqrt(sum((x1-x2).^2,2)),data.eyePos,data.(comppos),'un',0);
distVal = cellfun(@(x1,x2) x1(x2),ad,tIdx,'un',0);
szVal = cellfun(@(x1,x2) x1(x2),data.eyeSz,tIdx,'un',0);
if sign(alignto) == 1,      
    distVal = cellfun(@(x1) [x1(1:min(length(x1),alignto));nan(alignto - length(x1),1)],distVal,'un',0);
    szVal = cellfun(@(x1) [x1(1:min(length(x1),alignto));nan(alignto - length(x1),1)],szVal,'un',0);
elseif sign(alignto) == -1, 
    distVal = cellfun(@(x1) [nan(-alignto - length(x1),1);x1(end-min(length(x1),-alignto)+1:end)],distVal,'un',0);
    szVal = cellfun(@(x1) [nan(-alignto - length(x1),1);x1(end-min(length(x1),-alignto)+1:end)],szVal,'un',0);
end
distVal = cell2mat(distVal)';
szVal = cell2mat(szVal)';

end