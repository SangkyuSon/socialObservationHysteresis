function out = infstd(data,dim)
if nargin < 2, dim = 1; end

data(data==Inf) = nan;
out = std(data,dim,'omitnan');

end