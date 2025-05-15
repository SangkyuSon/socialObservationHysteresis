function [p,selModel] = fitpoly(x,y,deg)
if size(y,2)==1, y = y'; end
if size(x,2)==1, x = x'; end
if nargin < 3, deg = [1,10]; end
mindeg = deg(1);
maxdeg = deg(2);

N = length(y);
xx = linspace(min(x),max(x),100);
AICc = nan(maxdeg,1);
for d = mindeg:maxdeg,
    K = d + 1;
    p{d} = polyfit(x,y,d);
    yhat = polyval(p{d},x);
    SSE = sum((y-yhat).^2);
    AICc(d) = N*log(SSE/N) + 2*K + (2*K*(K+1))/(N-K-1);
end

[~,selModel] = min(AICc);
p = p{selModel};


end
