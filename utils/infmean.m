function out = infmean(data,dim)
if nargin < 2, dim = 1; end

data(data==Inf) = nan;
data(data==-Inf) = nan;
out = mean(data,dim,'omitnan');

end