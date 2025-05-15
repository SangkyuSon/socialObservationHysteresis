function xbar_custom(x,y,varargin)

if size(x,1) == 1, x = x'; end
if size(y,1) == 1, y = y'; end

options = struct(...
    'Color',[0,0,0],...
    'LineWidth',3,...
    'scatter',  1,...
    'scatterColor',[0,0,0],...
    'scatterAlpha',0.2,...
    'std',      0,...
    'unityLine',1);
options = checkOptions(options,varargin{:});

[xmu,xci] = sem(x,'no',size(x,1),'std',options.std);
[ymu,yci] = sem(y,'no',size(y,1),'std',options.std);

if options.scatter,
    scatter(x,y,50,options.scatterColor,'filled','MarkerFaceAlpha',options.scatterAlpha)
end
line([xmu-xci,xmu+xci],[ymu,ymu],'Color',options.Color,'LineWidth',options.LineWidth);
line([xmu,xmu],[ymu-yci,ymu+yci],'Color',options.Color,'LineWidth',options.LineWidth);

if options.unityLine,
    %line([min(x),max(x)]*1.1,[min(y),max(y)]*1.1,'Color',[1,1,1]*0.5,'LineStyle','--');
    line([min([x;y]),max([x;y])],[min([x;y]),max([x;y])],'Color',[1,1,1]*0.5,'LineStyle','--');
end



end