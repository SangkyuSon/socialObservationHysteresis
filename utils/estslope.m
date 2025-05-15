function [slo,itc,rsq] = estslope(Y,x)

[n,nT] = size(Y);
if nargin < 2, x = 1:nT; end
for k = 1:n,
    [co(k,:),dev(k,1)] = glmfit(x,Y(k,:)');
end

nulldev = sum((Y-mean(Y,2,'omitnan')).^2,2,'omitnan');
rsq = 1-dev./nulldev;

slo = co(:,2);
itc = co(:,1);

end