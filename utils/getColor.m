function col = getColor(nstep,col1,col2)

if nargin < 2,
    col1 = [0.8941, 0.1020, 0.1098];
    col2 = [0.2157, 0.4941, 0.7216];
end

for i = 1:3,
    col(:,i) = linspace(col1(i),col2(i),nstep);
end

col = mat2cell(col,ones(nstep,1),3);

end