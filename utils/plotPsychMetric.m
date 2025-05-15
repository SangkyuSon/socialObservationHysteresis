function out = plotPsychMetric(x,y,col,showRaw,lty)

if nargin < 3, col = [0,0,0]; end
if nargin < 4, showRaw = 1; end
if nargin < 5, lty = '-'; end

if showRaw,
    binno = 10;
    [by,bx] = histvec(y,x,binno+1);
    plot(bx,by,'Color',col,'LineStyle','--')
end
hold on
out = psychMetric(x,y,'findy',0.5);
plot(out.fitx,out.fity,'Color',col,'LineStyle',lty)
out.biny = by;
out.binx = bx;

end