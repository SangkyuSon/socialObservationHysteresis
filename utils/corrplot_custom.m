function [rho,p] = corrplot_custom(x,y,varargin)

if nargin<2, xx=x; x = xx(:,1); y = xx(:,2); end
if size(x,1)==1, x = x'; end
if size(y,1)==1, y = y'; end


tf = ishold;
if ~tf, cla; end

options = struct(...
    'method','Spearman',...
    'Color',[1,1,1]*0,...
    'alpha',0.3,...
    'regIntercept','on',...
    'regxlim',[min(x),max(x)],...
    'size',10,...
    'verbose', 1,...
    'regression',1);
options = checkOptions(options,varargin{:});
     
scatter(x,y,options.size,options.Color,'filled','MarkerFaceAlpha',options.alpha)
hold on

y(isinf(y)) = nan;
coef = glmfit(x,y,'normal','Constant',options.regIntercept);
coef = [flipud(coef);zeros(2-length(coef),1)];
if options.regression, 
    xx = linspace(options.regxlim(1),options.regxlim(2),100);
    plot(xx,xx*coef(1)+coef(2),'Color',options.Color); 
end

[rho,p] = corr(x,y,'type','Spearman','rows','pairwise');
if p < 0.05, if rho > 0, stringCol = [1,0,0]; elseif rho < 0, stringCol = [0,0,1]; end; else, stringCol = [0,0,0]; end
if options.verbose, text(nanmedian(x),nanmedian(y),sprintf('%s: %.3f (%.3f)',options.method(1),rho,p),'Color',stringCol); end

if tf, hold on; else hold off; end

end