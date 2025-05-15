function [y,x] = histvec(vec,idx,binno)

if size(vec,1)==1, vec = vec'; end
if nargin < 3,binno = 7; end

if length(binno)~=1, binval = binno; binno = length(binval); 
elseif binno == 0,   binval = [unique(idx(~isnan(idx)))-1e-3;max(idx)+1e-3]; binno = length(binval);
else,                binval = linspace(min(idx),max(idx),binno); end

[cnt, ed, bins] = histcounts(idx, binval);
ed = movmean(ed,2,'omitnan');
x = ed(2:end);

for b = 1:binno-1,
    y{b} = nanmean(vec(bins==b,:));
end

y = alignTimeseries(y,max(cellfun(@length,y)),1);

end