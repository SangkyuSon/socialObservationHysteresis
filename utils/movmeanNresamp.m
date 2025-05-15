function out = movmeanNresamp(data,winSz,dim,ismovsum)
if nargin < 4, ismovsum = 0; end

if winSz ~= 1,
    if nargin < 3, dim = 2; end
    
    if dim == 1, data = data'; end

    if ~ismovsum, tmp = movmean(data,winSz,2,'omitnan');
    else,         tmp = movsum(data,winSz,2,'omitnan');
    end
    out = tmp(:,floor(winSz/2)+1:winSz:end,:,:,:,:);

    if dim == 1, out = out'; end
else
    out = data;
end

end