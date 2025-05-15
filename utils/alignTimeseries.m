function out = alignTimeseries(data,len,alignto)
n = length(data);
if iscell(alignto), alignto = cell2mat(alignto); end
if nargin < 3, alignto = 1; end
if length(len) == 1, win = [-len:len]; else, win = [len(1):len(2)]; end
if length(alignto)==1, alignto = ones(1,n)*alignto; end
if size(data,1)==1, data = data'; end

sz = cell2mat(cellfun(@size,data,'un',0));
if isempty(alignto),
    out = nan(1,len*2+1);
    return
end

if mean(sz(logical(~sum(sz==0,2)),1) > sz(logical(~sum(sz==0,2)),2)), data = cellfun(@transpose,data,'un',0); end
sz = cell2mat(cellfun(@size,data,'un',0));
nv = setdiff(unique(sz(:,1)),0);

for k = 1:n,
    kdata = data{k};
    [knv,knt] = size(kdata);
    if knv == 0, knt = 0; end
    
    kwin = win+alignto(k);
    befsel = kwin<1;
    aftsel = kwin>knt;
    kwin(befsel | aftsel) = [];
    
    if alignto(k)==1,      out(k,:,:) = [kdata(:,1:min(knt,len)),nan(nv,len-knt)];
    elseif alignto(k)==-1, out(k,:,:) = [nan(nv,len-knt),kdata(:,end-min(knt,len)+1:end)]; 
    else,                  out(k,:,:) = [nan(nv,sum(befsel)),kdata(:,kwin),nan(nv,sum(aftsel))]; end
    
end

out = permute(out,[1,3,2]);

end