function [mu,ci,btmu] = sem(data,varargin)
% [mu,ci] = sem(data,varargin);
%        varargin :: 
%             'no',      number of subject
%             'nanVer',  use nanmean or mean?
options = struct('no',      min(size(data)),  ...
                 'nanVer',  1,...
                 'oriVer',  0,...
                 'std',     0);
options = checkOptions(options,varargin{:});

%% mu & ci
data = squeeze(data);
if length(size(data))~=2, error('need to provide data in 2 dimentional form'); end
if size(data,1)~=options.no, data = permute(data,[2,1]); end
%data(data==Inf) = [];

if options.nanVer,
    mu = squeeze(infmean(data,1));
    sd = squeeze(infstd(data,1));
elseif options.nanVer == 0,
    mu = squeeze(mean(data,1));
    sd = squeeze(std(data,[],1));
end

if options.oriVer==1,
    mu = rad2deg(circ_mean(deg2rad(data)*2,[],1))/2;
    sd = rad2deg(circ_std(deg2rad(data)*2,[],0,1)')'/2;
    %mu = circ_mean_ori(data,[],1)';
    %sd = circ_std_ori(data)';
elseif options.oriVer == 2, % direction version
    mu = rad2deg(circ_mean(deg2rad(data),[],1)')';
    sd = rad2deg(circ_std(deg2rad(data),[],0,1)')';
end

btmu = [];
if options.std==0, ci = 1.*sd./sqrt(options.no); 
elseif options.std==1, ci = 1.96*sd; 
elseif options.std==2, ci = sd;
elseif options.std==3, ci = [quantile(data,0.025)-mu;quantile(data,0.975)-mu];
elseif options.std==4, ci = [quantile(data,0.25)-mu;quantile(data,0.75)-mu]; 
elseif options.std>=5, % bootstrap
    bootNo = 1000;
    clear btmu
    for bt = 1:bootNo,
        btIdx = ceil(rand(1,options.no)*options.no);
        btmu(bt,:) = nanmean(data(btIdx,:),1);
    end
    if options.std == 5,
        ci = [quantile(btmu,0.025)-mu;quantile(btmu,0.975)-mu];
    elseif options.std == 6,
        ci = [quantile(btmu,0.005)-mu;quantile(btmu,0.995)-mu];
    elseif options.std == 7,
        ci = [quantile(btmu,0.0005)-mu;quantile(btmu,0.9995)-mu];
    end
end

end