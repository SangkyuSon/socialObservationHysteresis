function data = transferVal(data,idx,val)

if isnan(idx), idx = isnan(data); end
if length(idx)==1, idx = data==idx; end

%idx = boolean(idx);

if ~isempty(val),  data(idx) = val;
else,              data(idx) = [];
end

if isempty(idx) & isempty(data), data = val;end 

end